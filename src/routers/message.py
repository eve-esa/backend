from src.schemas.message import MessageUpdate, CreateMessageResponse
from src.services.generate_answer import (
    GenerationRequest,
    generate_answer,
    maybe_rollup_and_trim_history,
)
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.models.collection import Collection as CollectionModel
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from src.database.models.user import User
from src.middlewares.auth import get_current_user
import logging
from src.services.generate_answer import generate_answer_json_stream_generator
from src.utils.helpers import extract_document_data, extract_year_range_from_filters

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/conversations/{conversation_id}/messages", response_model=CreateMessageResponse
)
async def create_message(
    request: GenerationRequest,
    conversation_id: str,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
):
    message = None
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        # lookup query to check if some of the collection ids from other users are in the request.collection_ids
        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )

        if len(other_users_collections) > 0:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        request.collection_ids = request.collection_ids + request.public_collections

        # All user collections are used by default
        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        request.private_collections_map = {c.id: c.name for c in user_collections}
        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]
        # remove "Wiley AI Gateway" from collection_ids
        request.collection_ids = [
            c for c in request.collection_ids if c != "Wiley AI Gateway"
        ]
        logger.info(f"Collection IDs: {request.collection_ids}")

        # Extract year range from filters for MCP usage
        try:
            request.year = extract_year_range_from_filters(request.filters)
        except Exception:
            request.year = None

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output="",
            documents=[],
            use_rag=False,
            request_input=request,
            metadata={},
        )

        answer, results, is_rag, loop_result, latencies, prompts = (
            await generate_answer(request, conversation_id=conversation_id)
        )

        documents_data = []
        if results:
            documents_data = [extract_document_data(result) for result in results]

        message.output = answer
        message.documents = documents_data
        message.use_rag = is_rag
        existing_metadata = dict(getattr(message, "metadata", {}) or {})
        existing_metadata.update(
            {
                "latencies": latencies,
                "prompts": prompts,
            }
        )
        message.metadata = existing_metadata
        await message.save()

        # Schedule rollup as background task to avoid blocking response
        background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)

        return {
            "id": message.id,
            "query": request.query,
            "answer": answer,
            "documents": documents_data,
            "use_rag": is_rag,
            "conversation_id": conversation_id,
            "loop_result": loop_result,
            "collection_ids": request.collection_ids,
            "metadata": {
                "latencies": latencies,
            },
        }
    except HTTPException as http_exc:
        if message:
            try:
                message.metadata = {"error": str(getattr(http_exc, "detail", http_exc))}
                await message.save()
            except Exception:
                pass
        raise
    except Exception as e:
        if message:
            try:
                message.metadata = {"error": str(e)}
                await message.save()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/conversations/{conversation_id}/messages/{message_id}/retry")
async def retry(
    conversation_id: str,
    message_id: str,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
):
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        message = await Message.find_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        if message.conversation_id != conversation_id:
            raise HTTPException(
                status_code=404, detail="Message not found in this conversation"
            )

        if not message.request_input:
            raise HTTPException(
                status_code=400,
                detail="This message cannot be retried",
            )

        answer, results, is_rag, loop_result, latencies, prompts = (
            await generate_answer(
                message.request_input, conversation_id=conversation_id
            )
        )

        documents_data = []
        if results:
            documents_data = [extract_document_data(result) for result in results]

        message.output = answer
        message.documents = documents_data
        message.use_rag = is_rag
        existing_metadata = dict(getattr(message, "metadata", {}) or {})
        existing_metadata.update({"latencies": latencies, "prompts": prompts})
        message.metadata = existing_metadata
        await message.save()

        # Schedule rollup as background task to avoid blocking response
        background_tasks.add_task(maybe_rollup_and_trim_history, conversation_id)

        return {
            "id": message.id,
            "query": message.input,
            "answer": answer,
            "documents": documents_data,
            "use_rag": is_rag,
            "conversation_id": conversation_id,
            "loop_result": loop_result,
            "collection_ids": message.request_input.collection_ids,
            "metadata": {
                "latencies": latencies,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.patch("/conversations/{conversation_id}/messages/{message_id}")
async def update_message(
    conversation_id: str,
    message_id: str,
    request: MessageUpdate,
    requesting_user: User = Depends(get_current_user),
):
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        message = await Message.find_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        if message.conversation_id != conversation_id:
            raise HTTPException(
                status_code=404, detail="Message not found in this conversation"
            )

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to update feedback for this message",
            )

        if request.feedback is not None:
            message.feedback = request.feedback.value

        if request.was_copied is not None:
            message.was_copied = request.was_copied

        if request.feedback_reason is not None:
            message.feedback_reason = request.feedback_reason

        await message.save()

        return {"message": "Feedback updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post(
    "/conversations/{conversation_id}/stream_messages",
    response_class=StreamingResponse,
)
async def create_message_stream(
    request: GenerationRequest,
    conversation_id: str,
    background_tasks: BackgroundTasks,
    requesting_user: User = Depends(get_current_user),
):
    message = None
    try:
        conversation = await Conversation.find_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != requesting_user.id:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to add a message to this conversation",
            )

        # lookup query to check if some of the collection ids from other users are in the request.collection_ids
        other_users_collections = await CollectionModel.find_all(
            filter_dict={
                "id": {"$in": request.public_collections},
                "user_id": {"$ne": requesting_user.id},
            }
        )

        if len(other_users_collections) > 0:
            raise HTTPException(
                status_code=403,
                detail="You are not allowed to use collections from other users",
            )

        request.collection_ids = request.collection_ids + request.public_collections

        # All user collections are used by default
        user_collections = await CollectionModel.find_all(
            filter_dict={"user_id": requesting_user.id}
        )

        request.private_collections_map = {c.id: c.name for c in user_collections}
        if len(user_collections) > 0:
            request.collection_ids = request.collection_ids + [
                c.id for c in user_collections
            ]
        # remove "Wiley AI Gateway" from collection_ids
        request.collection_ids = [
            c for c in request.collection_ids if c != "Wiley AI Gateway"
        ]
        logger.info(f"Collection IDs: {request.collection_ids}")

        # Extract year range from filters for MCP usage
        try:
            request.year = extract_year_range_from_filters(request.filters)
        except Exception:
            request.year = None

        message = await Message.create(
            conversation_id=conversation_id,
            input=request.query,
            output="",
            documents=[],
            use_rag=False,
            request_input=request,
            metadata={},
        )

        response = StreamingResponse(
            generate_answer_json_stream_generator(
                request,
                conversation_id=conversation_id,
                message_id=message.id,
                background_tasks=background_tasks,
            ),
            media_type="text/event-stream",
        )
        # Set SSE-friendly headers to prevent proxy/client reconnect loops
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"  # Nginx buffering off if present
        return response
    except HTTPException as http_exc:
        if message:
            try:
                message.metadata = {"error": str(getattr(http_exc, "detail", http_exc))}
                await message.save()
            except Exception:
                pass
        raise
    except Exception as e:
        if message:
            try:
                message.metadata = {"error": str(e)}
                await message.save()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
