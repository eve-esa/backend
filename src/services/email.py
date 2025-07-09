"""
Email service using Python's built-in smtplib and email modules with Jinja2 templating.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from src.config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    EMAIL_FROM_ADDRESS,
    EMAIL_FROM_NAME,
)

logger = logging.getLogger(__name__)


class EmailSendError(Exception):
    """Exception raised when email sending fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class EmailService:
    """Service for sending transactional emails using SMTP with Jinja2 templating."""

    def __init__(self):
        """Initialize the email service with Jinja2 environment."""
        self.smtp_host = SMTP_HOST
        self.smtp_port = SMTP_PORT
        self.smtp_username = SMTP_USERNAME
        self.smtp_password = SMTP_PASSWORD
        self.from_email = EMAIL_FROM_ADDRESS
        self.from_name = EMAIL_FROM_NAME

        # Set up Jinja2 environment
        templates_dir = Path(__file__).parent.parent.parent / "templates" / "emails"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render an email template with the given context using Jinja2.

        Args:
            template_name: Name of the template file (e.g., 'forgot_password.html')
            context: Variables to pass to the template

        Returns:
            Rendered HTML content

        Raises:
            EmailSendError: If template rendering fails
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)

        except TemplateNotFound as e:
            logger.error(f"Template not found: {template_name}")
            raise EmailSendError(f"Template not found: {template_name}") from e
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {str(e)}")
            raise EmailSendError(
                f"Failed to render template {template_name}: {str(e)}"
            ) from e

    def send_email(
        self, to_email: str, subject: str, template_name: str, context: Dict[str, Any]
    ):
        try:
            html_content = self._render_template(template_name, context)
            self._send_email(to_email, subject, html_content)

        except EmailSendError:
            raise
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            raise EmailSendError(f"Failed to send email to {to_email}: {str(e)}") from e

    def _send_email(self, to_email: str, subject: str, html_content: str):
        """
        Send an email using SMTP.
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = to_email

            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_username and self.smtp_password:
                    server.starttls()
                    server.login(self.smtp_username, self.smtp_password)

                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            raise EmailSendError(f"Failed to send email to {to_email}: {str(e)}") from e


email_service = EmailService()
