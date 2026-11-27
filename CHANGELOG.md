# Changelog

Notable changes to the EVE backend. Newest first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From v0.0.8 onward this file is written by [release-please](https://github.com/googleapis/release-please)
from the pull request titles merged into `main`, so entries are not added by hand. Work that has landed
but is not yet released is visible in the open `chore(main): release ...` pull request, which is also
what cuts the release: merging it commits the version, creates the `vX.Y.Z` tag, publishes the GitHub
Release and promotes staging. Production is promoted from there by an explicit dispatch.

## [1.9.0](https://github.com/eve-esa/backend/compare/v1.8.1...v1.9.0) (2026-11-27)


### Added

* **conversations:** report the turn still running ([#412](https://github.com/eve-esa/backend/issues/412)) ([766b57c](https://github.com/eve-esa/backend/commit/766b57c8a55753c20d2b3c02570ffa502db2f0f6))
* **mcp:** let users bring their own toolkit credentials ([#410](https://github.com/eve-esa/backend/issues/410)) ([94d0bb8](https://github.com/eve-esa/backend/commit/94d0bb804e09dc65ef18e60189f2594c229e456d))
* **mcp:** store registry environments ([#416](https://github.com/eve-esa/backend/issues/416)) ([9b4100d](https://github.com/eve-esa/backend/commit/9b4100df86d54339ff5bbbcf714e5394a3fe4a97))
* **mcp:** store registry environments ([#416](https://github.com/eve-esa/backend/issues/416)) ([c4ed06b](https://github.com/eve-esa/backend/commit/c4ed06b706ae126f0b8b37676164ce657854c268))
* refuse writes past the storage cap ([#415](https://github.com/eve-esa/backend/issues/415)) ([30f77b2](https://github.com/eve-esa/backend/commit/30f77b29b486380d0bf78adfc90cdea88b4c1129))
* stream agent reasoning ([#417](https://github.com/eve-esa/backend/issues/417)) ([55afc78](https://github.com/eve-esa/backend/commit/55afc78290dbfefb5c21356e794debfdadd08dbb))


### Fixed

* **agents:** close parenthetical link labels ([#409](https://github.com/eve-esa/backend/issues/409)) ([24724f8](https://github.com/eve-esa/backend/commit/24724f85fe98e76e0d328c08777233f49489e6a2))

## [1.8.1](https://github.com/eve-esa/backend/compare/v1.8.0...v1.8.1) (2026-11-20)


### Fixed

* **users:** read the caller from the primary on /users/me ([#407](https://github.com/eve-esa/backend/issues/407)) ([e03ca9a](https://github.com/eve-esa/backend/commit/e03ca9a746498632ed356fdc7a4c796a5407668e))

## [1.8.0](https://github.com/eve-esa/backend/compare/v1.7.0...v1.8.0) (2026-11-20)


### Added

* **agents:** add a conversation's documents to its retrieval ([#362](https://github.com/eve-esa/backend/issues/362)) ([e15e365](https://github.com/eve-esa/backend/commit/e15e365e3555efc082f50ceeafda2a1cca6d17e8)) (flag FEATURE_CONVERSATION_DOCUMENTS, prod off)
* **agents:** add sandbox display tool ([#363](https://github.com/eve-esa/backend/issues/363)) ([5aac452](https://github.com/eve-esa/backend/commit/5aac452f912a6dd0c2017bbd12a82fd5b798287e)) (SANDBOX_PROVIDER, prod off)
* **agents:** enrich the agentic stream frames with step ids, status and timing ([73fe31c](https://github.com/eve-esa/backend/commit/73fe31c413898cf74b90ec2de2cc0d3d6bf6cd1b))
* **agents:** mirror MCP files in sandbox ([#366](https://github.com/eve-esa/backend/issues/366)) ([70ee664](https://github.com/eve-esa/backend/commit/70ee664db8dbad0d806c70d5e6deaba1b057f608)) (SANDBOX_PROVIDER, prod off)
* **agents:** put the endpoint and token usage on the final frame ([b3e223d](https://github.com/eve-esa/backend/commit/b3e223daa52d2802af1cf97f5650f1b87e552c28))
* **agents:** save the sandbox workspace ([#379](https://github.com/eve-esa/backend/issues/379)) ([52b5176](https://github.com/eve-esa/backend/commit/52b517627f880246002b2539d5ecba5bae9c84f8)) (SANDBOX_PROVIDER, prod off)
* **agents:** search stored conversation history ([#359](https://github.com/eve-esa/backend/issues/359)) ([4780f96](https://github.com/eve-esa/backend/commit/4780f967f5d75c8a97f22c35cfc824f104a61f2d)) (AGENT_GRAPH_TYPE=deep, prod off)
* **agents:** store offloaded tool results in s3 ([#358](https://github.com/eve-esa/backend/issues/358)) ([fb1f012](https://github.com/eve-esa/backend/commit/fb1f0126b10fcdf5e0023a01e4fae2f04e12d86a)) (AGENT_GRAPH_TYPE=deep, prod off)
* **agents:** stream tool call arguments ([#365](https://github.com/eve-esa/backend/issues/365)) ([0c98821](https://github.com/eve-esa/backend/commit/0c988213ed7967b157f1fb879d87b3fd60102ea4))
* **agents:** summarize deep history ([#360](https://github.com/eve-esa/backend/issues/360)) ([132dadd](https://github.com/eve-esa/backend/commit/132dadd9ec3c4ab4c576f17c1b84e2eacc4bc980)) (AGENT_GRAPH_TYPE=deep, prod off)
* **agents:** support the deep agent and sandbox skills ([#350](https://github.com/eve-esa/backend/issues/350)) ([3ce45de](https://github.com/eve-esa/backend/commit/3ce45de68132e68d64d9a74fe2cb935927b6e9d4)) (AGENT_GRAPH_TYPE=deep, prod off)
* **agui:** AG-UI 1.0 route for standard clients (flag FEATURE_AGUI_STREAM, prod off) ([bd15b43](https://github.com/eve-esa/backend/commit/bd15b43779d47a5d42a2feabd146252a6ad2637e))
* **agui:** translate legacy stream frames to AG-UI 1.0 events ([494efb8](https://github.com/eve-esa/backend/commit/494efb85d4dbf8cee31b44ea9568c5e1853db25d)) (flag FEATURE_AGUI_STREAM, prod off)
* **api:** add read-only files and sandbox API ([#384](https://github.com/eve-esa/backend/issues/384)) ([771abac](https://github.com/eve-esa/backend/commit/771abacfdcec4596d334272ab85514a771ae8740)) (flag FEATURE_SANDBOX_API, prod off)
* **artifacts:** give attached files to the agent ([daf2fc8](https://github.com/eve-esa/backend/commit/daf2fc8a23f3366ca2483c3a0878fc7810d193b3)) (flag FEATURE_ARTIFACT_UPLOADS, prod off)
* **citations:** inline citations on agentic answers (flag FEATURE_INLINE_CITATIONS, prod off) ([64657c4](https://github.com/eve-esa/backend/commit/64657c4d5db46b5aed9a03f1ae1ae414a216ec2f))
* **citations:** name each source's provider with a slug ([8d6115d](https://github.com/eve-esa/backend/commit/8d6115d151b39f587cbddb9765327bcd0a60ea5a))
* **documents:** attach documents to one conversation ([#333](https://github.com/eve-esa/backend/issues/333)) ([f59f5d0](https://github.com/eve-esa/backend/commit/f59f5d0780cd2eca51e598c78de79434c35871e2)) (flag FEATURE_CONVERSATION_DOCUMENTS, prod off)
* **documents:** list and delete a conversation's documents ([#356](https://github.com/eve-esa/backend/issues/356)) ([a96a871](https://github.com/eve-esa/backend/commit/a96a8718f239a4f11add2b7f2d45a1cc81b5eef6)) (flag FEATURE_CONVERSATION_DOCUMENTS, prod off)
* **mail:** load templates from a published pack ([#380](https://github.com/eve-esa/backend/issues/380)) ([3d743ad](https://github.com/eve-esa/backend/commit/3d743adf16e69d3291d65ba2f10158e633698e1c))
* **mail:** mail listed addresses without an account ([#382](https://github.com/eve-esa/backend/issues/382)) ([eb65dfa](https://github.com/eve-esa/backend/commit/eb65dfa80960ed5293168031781674b1462f3896))
* **mcp:** return server attribution ([#354](https://github.com/eve-esa/backend/issues/354)) ([c8ed480](https://github.com/eve-esa/backend/commit/c8ed480704c7f6a0cf637c8425490d880d051361))
* **models:** add Mistral Large 4 as a selectable JSC model ([#342](https://github.com/eve-esa/backend/issues/342)) ([75472b1](https://github.com/eve-esa/backend/commit/75472b16f2c0eea396bc231935e05cdb9aff6df7))
* **sandbox:** keep a sandbox per conversation ([#353](https://github.com/eve-esa/backend/issues/353)) ([b550a81](https://github.com/eve-esa/backend/commit/b550a81b61c4541b205111cf8f08284592709c21)) (SANDBOX_PROVIDER, prod off)


### Fixed

* **agents:** decide the skills reload from the recorded key ([f43729a](https://github.com/eve-esa/backend/commit/f43729a3885dab509803841153edb0280d64bd52))
* **agents:** do not describe an image the model cannot view ([#393](https://github.com/eve-esa/backend/issues/393)) ([87cad55](https://github.com/eve-esa/backend/commit/87cad55c73b415c8119e0a41cfec722fe9257c70))
* **agents:** do not offer to re-show an uploaded image ([#400](https://github.com/eve-esa/backend/issues/400)) ([d518624](https://github.com/eve-esa/backend/commit/d51862496d0c3218ccae1da682e423c13952763f))
* **agents:** log a gone sandbox and readiness retries at INFO ([e52558a](https://github.com/eve-esa/backend/commit/e52558acc06d390f19ec888de8307db6d413459b))
* **agents:** log a sandbox command cut by a user Stop at INFO ([70abb1c](https://github.com/eve-esa/backend/commit/70abb1cc02d86f0d62064d9e3d4ed153d4d5a28e))
* **agents:** paragraph break on every model call, topped up, without blank or joined parts ([979d28a](https://github.com/eve-esa/backend/commit/979d28a84600c479ba378a594d77659fcb346e34))
* **agents:** persist cut tool calls on Stop and give retries their own id ([6860f7c](https://github.com/eve-esa/backend/commit/6860f7c9e76452a3f33524a58e0f103f7087a14c))
* **agents:** record an empty agentic answer as empty_answer ([98f602b](https://github.com/eve-esa/backend/commit/98f602b0dba675a829864ae8a67940eb77e5c55a))
* **agents:** reload skills after a failed checkpoint read or listing ([e6569ea](https://github.com/eve-esa/backend/commit/e6569ea7d90c1b42cb97a92d598ae7c455ec6696))
* **agents:** reload skills when the recorded key is missing ([31ac20f](https://github.com/eve-esa/backend/commit/31ac20f42ffde7155ce5f076eb3d4a5997309df6))
* **agents:** reload skills when the set changes mid-conversation ([b5db0d5](https://github.com/eve-esa/backend/commit/b5db0d5be78ea6c293d55a0d5d085251b0faf22b))
* **agents:** report an empty agentic answer the same way on every path ([59bb041](https://github.com/eve-esa/backend/commit/59bb041fbae84e9d1694efda78a114f052750b52))
* **agents:** restore a dropped link bracket ([#403](https://github.com/eve-esa/backend/issues/403)) ([531008e](https://github.com/eve-esa/backend/commit/531008e8dc5e07c536fb9afe192be8dfbb4fa761))
* **agents:** separate the answer text of two model calls with a paragraph break ([03f979e](https://github.com/eve-esa/backend/commit/03f979ec79f7d5bf51b24c24c12dd748f37ad621))
* **agents:** show images the agent makes in the answer ([#396](https://github.com/eve-esa/backend/issues/396)) ([259f4f5](https://github.com/eve-esa/backend/commit/259f4f5cd68d8150f2b2dcc354f7e0c6e218b6fe))
* **agents:** stop a gone sandbox and keep retry warnings outside attach ([9aadf4e](https://github.com/eve-esa/backend/commit/9aadf4e20d0118efcf18b010444f40906d72d633))
* **agents:** stop the agent from repeating the same tool call ([#401](https://github.com/eve-esa/backend/issues/401)) ([204d227](https://github.com/eve-esa/backend/commit/204d227c30b8a0593f98c8ec47f76a5d6aff41fe))
* **agents:** tell the agent a platform model cannot read images ([#389](https://github.com/eve-esa/backend/issues/389)) ([7a8fe04](https://github.com/eve-esa/backend/commit/7a8fe0427c88f655e8295f2b89f301e8d85605a9))
* **agents:** tell the model it can show an image it cannot see ([#399](https://github.com/eve-esa/backend/issues/399)) ([349bc18](https://github.com/eve-esa/backend/commit/349bc18c875995723bdeee767bced171abf01cd7))
* **agui:** end every run with one terminal event and claim run ids across workers ([d490411](https://github.com/eve-esa/backend/commit/d490411f4566c1ba1ab6e8d6c142dc3145883ad4))
* **agui:** keep the stream encodable and tell a dropped client from a Stop ([543b1bf](https://github.com/eve-esa/backend/commit/543b1bf14b683035fa920d4b88bd530024a4380f))
* **agui:** let a standard AG-UI client stop runs and read tool calls ([db28f42](https://github.com/eve-esa/backend/commit/db28f42f68cccb9000f3e7a418ada56b3e2ac790))
* **artifacts:** clean upload filenames and serve tool PDFs as attachments ([adab787](https://github.com/eve-esa/backend/commit/adab78767ff882235f45cbe9cea0fc16b854ce8e))
* **artifacts:** close the upload route where uploads are off ([f289131](https://github.com/eve-esa/backend/commit/f289131ccd081bbd5d39623ea79a5b8624e62762))
* **artifacts:** erase files before the conversation and answer 503 when incomplete ([8818bf9](https://github.com/eve-esa/backend/commit/8818bf9c10b6e9d88722494627be3aa97d65e788))
* **artifacts:** erase files with their conversation and on owner delete ([5827fcb](https://github.com/eve-esa/backend/commit/5827fcbd8f649c16987ca6eacc4254310e5298a2))
* **artifacts:** keep an attached file's row on a key that exists ([#395](https://github.com/eve-esa/backend/issues/395)) ([c4ab0d0](https://github.com/eve-esa/backend/commit/c4ab0d0f9f8513bf1c85124115a8451e9d7bdbfd))
* **artifacts:** render a tool image the answer links by its filename ([#386](https://github.com/eve-esa/backend/issues/386)) ([93f1e04](https://github.com/eve-esa/backend/commit/93f1e0454a4824bdc18a284ddb5230fd8369e010))
* **artifacts:** rewrite any link that carries a turn artifact id ([#390](https://github.com/eve-esa/backend/issues/390)) ([c30d9e5](https://github.com/eve-esa/backend/commit/c30d9e51f2ad118981c0e6e22009957cc863577b))
* **artifacts:** serve artifact files safely ([669cfd6](https://github.com/eve-esa/backend/commit/669cfd67655a90e2d17f8219520292151ce2b09b))
* **artifacts:** store tool files with the type their bytes show ([89eb5a0](https://github.com/eve-esa/backend/commit/89eb5a0a9f2b5ca641f633d7f01db30185c8c3e8))
* **chat:** stop the running turn before a retry ([#364](https://github.com/eve-esa/backend/issues/364)) ([d6326b7](https://github.com/eve-esa/backend/commit/d6326b716258ad9f01c94d952e2fedfe520576f3))
* **citations:** conversation handles, stripped raw handles and held text ([7a6a271](https://github.com/eve-esa/backend/commit/7a6a271bfdc6876663ad290903c0be0680fabfe7))
* **citations:** keep model brackets and offsets ([b16266a](https://github.com/eve-esa/backend/commit/b16266a093c277fdf0de90b85bd3df7181a9b45d))
* **citations:** keep the retrieve tool call id, drop chunk suffixes, plain passages ([65dc07e](https://github.com/eve-esa/backend/commit/65dc07eaf59e37774855e89cbfac39d1e90cbb79))
* **citations:** list each uncited document once in sources_retrieved ([fb280a2](https://github.com/eve-esa/backend/commit/fb280a2bb5174333a06faeb6b8ee31e64e0ab1f2))
* **conversations:** delete agent memory with the conversation ([#340](https://github.com/eve-esa/backend/issues/340)) ([b774176](https://github.com/eve-esa/backend/commit/b774176f4f3337fd9d37963b20670f9a6621e689))
* **deps:** pin agents to main e50e253 ([#391](https://github.com/eve-esa/backend/issues/391)) ([36fcc43](https://github.com/eve-esa/backend/commit/36fcc431772b33f27ba36a9d06a3df24fa6c1081))
* **deps:** pin agents to v0.5.6 ([#406](https://github.com/eve-esa/backend/issues/406)) ([62572e2](https://github.com/eve-esa/backend/commit/62572e24e92eda25ac973bdfccc3883e40f4ae58))
* **deps:** pymongo, langgraph and OpenTelemetry for the open alerts ([#344](https://github.com/eve-esa/backend/issues/344)) ([7ec77d5](https://github.com/eve-esa/backend/commit/7ec77d5c9e18f6cf273f6bd0e9e78552729640c6))
* **generation:** run turns without a conversation without memory ([#347](https://github.com/eve-esa/backend/issues/347)) ([45ee4fb](https://github.com/eve-esa/backend/commit/45ee4fb9760efa864600a3a89545f4a98532f4d1))
* **hallucination:** judge with the clock and the turn's tool results ([8a83ac0](https://github.com/eve-esa/backend/commit/8a83ac09e891f6b0045ffb4523ebbf58df7efb15))
* **hallucination:** keep tool evidence inside its block and judge on the answer's clock ([c0b09cc](https://github.com/eve-esa/backend/commit/c0b09ccbaf8701db89aaf342c02fc402616f9c64))
* **llm:** stop retrying JSC when Blablador rate limits a request ([d0865c8](https://github.com/eve-esa/backend/commit/d0865c87e87c371a856e37356d45dcdd36bc8d2e))
* **llm:** stop retrying JSC when Blablador rate limits a request ([#381](https://github.com/eve-esa/backend/issues/381)) ([d0865c8](https://github.com/eve-esa/backend/commit/d0865c87e87c371a856e37356d45dcdd36bc8d2e))
* **mcp:** forward the user token only to servers that need it ([26eb069](https://github.com/eve-esa/backend/commit/26eb0696d9178d9a039fae2f9d718adcc5df545d))
* **mcp:** resolve the caller's row first when a name is shared ([5431b5d](https://github.com/eve-esa/backend/commit/5431b5dfba2993ec95936248e0dd20740d9b2657))
* **observability:** do not count an expected missing S3 key as an error ([#397](https://github.com/eve-esa/backend/issues/397)) ([06254f9](https://github.com/eve-esa/backend/commit/06254f90dd14c10535dcf3710a1711733eb35bcc))
* restart agentic budget after tools ([#355](https://github.com/eve-esa/backend/issues/355)) ([820bd23](https://github.com/eve-esa/backend/commit/820bd234d7286a8ebdbd05f3000aea0fb23825c6))
* **sandbox:** let the model use the workspace path it sees ([#398](https://github.com/eve-esa/backend/issues/398)) ([3d711c3](https://github.com/eve-esa/backend/commit/3d711c357af96244995ba26464be6c01d6202b7c))
* **sandbox:** run commands from the workspace, quoted whole ([#388](https://github.com/eve-esa/backend/issues/388)) ([5b4b960](https://github.com/eve-esa/backend/commit/5b4b96023f774978853b1dfc56f35c7451ce01b8))
* **sandbox:** skip the snapshot of a gone sandbox ([62032e9](https://github.com/eve-esa/backend/commit/62032e91da828a397c55c4e6c5234952fcd421db))
* **sandbox:** stop the post-turn workspace save once the sandbox is gone ([#387](https://github.com/eve-esa/backend/issues/387)) ([13ee4a2](https://github.com/eve-esa/backend/commit/13ee4a244b0cdc50bf686cc3730614890c010cef))
* **sandbox:** tell the agent to run files by name, never through /workspace ([#392](https://github.com/eve-esa/backend/issues/392)) ([591d2dd](https://github.com/eve-esa/backend/commit/591d2dde1f8286b7219e057aadae49b76ba81309))
* **sandbox:** tell the model how to run a file it just wrote ([#394](https://github.com/eve-esa/backend/issues/394)) ([ea46525](https://github.com/eve-esa/backend/commit/ea46525d9f7f8ed4a795ef6301ea11755c3ab100))
* size each model's context window ([#405](https://github.com/eve-esa/backend/issues/405)) ([8792b23](https://github.com/eve-esa/backend/commit/8792b2387a39af6a9762342d3f9ac1b527f47a9e))
* **skills:** pass the skills path into compile ([#352](https://github.com/eve-esa/backend/issues/352)) ([03742d7](https://github.com/eve-esa/backend/commit/03742d7b96f89941de1c88c7c980b7c9739031cb))
* **stats:** log a missing CO2eq comparison set as a warning ([#385](https://github.com/eve-esa/backend/issues/385)) ([3451784](https://github.com/eve-esa/backend/commit/34517848db702c198203aabac84d18f414f5e622))
* **storage:** size the S3 pool for parallel reads and stop an unused tag ([#402](https://github.com/eve-esa/backend/issues/402)) ([2d824f5](https://github.com/eve-esa/backend/commit/2d824f5c77e66a435623d18d5a066880147eee44))
* **workspace:** apply artifact file rules ([2459731](https://github.com/eve-esa/backend/commit/2459731a4759d3af300ba8d3c52bfc56ac7988a7))


### Changed

* **agui:** call the stream translator an adapter ([3eb1b9e](https://github.com/eve-esa/backend/commit/3eb1b9e089639a84e290db7108ede9343fe5fbf0))

## [1.7.0](https://github.com/eve-esa/backend/compare/v1.6.0...v1.7.0) (2026-10-29)


### Added

* **documents:** reconcile private document counters ([#332](https://github.com/eve-esa/backend/issues/332)) ([b56bd60](https://github.com/eve-esa/backend/commit/b56bd600462375ddf6048c569b42f0b9a7e984a4))


### Fixed

* **artifacts:** retry the daily quota upsert once on a duplicate key ([#335](https://github.com/eve-esa/backend/issues/335)) ([161b1ab](https://github.com/eve-esa/backend/commit/161b1ab31dd73e2769beb06e858687125a4ac9f5))
* **retrieval:** switch Wiley to the Scholar Gateway v2 connector ([#337](https://github.com/eve-esa/backend/issues/337)) ([75f2b2d](https://github.com/eve-esa/backend/commit/75f2b2d512005ed5da2608c99b7752d041bc792b))

## [1.6.0](https://github.com/eve-esa/backend/compare/v1.5.0...v1.6.0) (2026-10-28)


### Added

* **mail:** one dark banner layout for every mail ([#310](https://github.com/eve-esa/backend/issues/310)) ([8c6aca1](https://github.com/eve-esa/backend/commit/8c6aca11c1a0d294def4f8fd8795ff90e3cbeba4))

## [1.5.0](https://github.com/eve-esa/backend/compare/v1.4.0...v1.5.0) (2026-10-15)


### Added

* **mail:** fall back to SES when SMTP delivery fails ([#323](https://github.com/eve-esa/backend/issues/323)) ([84642c5](https://github.com/eve-esa/backend/commit/84642c5f3bef9ed962c73d46008c3bd7a45c1b91))
* **mail:** route canary recipients through SMTP ([#321](https://github.com/eve-esa/backend/issues/321)) ([700d981](https://github.com/eve-esa/backend/commit/700d981567ca4926b5c0c99e651b23984354ebad))


### Fixed

* **mail:** greet the SMTP relay with the sender domain ([#324](https://github.com/eve-esa/backend/issues/324)) ([8b2f4c6](https://github.com/eve-esa/backend/commit/8b2f4c6509f4241bcc9be155cc5b7708021b6604))

## [1.4.0](https://github.com/eve-esa/backend/compare/v1.3.0...v1.4.0) (2026-10-15)


### Added

* unlimited token plan ([#319](https://github.com/eve-esa/backend/issues/319)) ([9576879](https://github.com/eve-esa/backend/commit/95768791266f9c880bae2d085c93aac11475f265))

## [1.3.0](https://github.com/eve-esa/backend/compare/v1.2.1...v1.3.0) (2026-10-14)


### Added

* rename token plans to env names ([#317](https://github.com/eve-esa/backend/issues/317)) ([fa2425f](https://github.com/eve-esa/backend/commit/fa2425f6a128cb18b86b9e18b1d56502896f56fc))

## [1.2.1](https://github.com/eve-esa/backend/compare/v1.2.0...v1.2.1) (2026-10-12)


### Fixed

* **deps:** pin agents to v0.5.5 ([#315](https://github.com/eve-esa/backend/issues/315)) ([dc1d295](https://github.com/eve-esa/backend/commit/dc1d2959571e06d5a8fbeae5832d3fc8bc1f0f05))

## [1.2.0](https://github.com/eve-esa/backend/compare/v1.1.0...v1.2.0) (2026-10-07)


### Added

* **api:** deprecate the classic message routes ([#307](https://github.com/eve-esa/backend/issues/307)) ([86193c1](https://github.com/eve-esa/backend/commit/86193c1fdaf5a29d04a599d19a697173b4f9c657))


### Fixed

* **agents:** retrieve first and drop unused tools in the react prompt ([#309](https://github.com/eve-esa/backend/issues/309)) ([eef9a88](https://github.com/eve-esa/backend/commit/eef9a88e948ee10bb646c7b7b5167466aa81c58b))

## [1.1.0](https://github.com/eve-esa/backend/compare/v1.0.3...v1.1.0) (2026-10-07)


### Added

* **retrieval:** apply every filter to all collections ([#305](https://github.com/eve-esa/backend/issues/305)) ([db9dc03](https://github.com/eve-esa/backend/commit/db9dc03788009f82a81943c998c8bace0854bae4))


### Fixed

* **agents:** send the tool query with the tool call event ([#303](https://github.com/eve-esa/backend/issues/303)) ([b405ceb](https://github.com/eve-esa/backend/commit/b405ceb71e66cb58e342772ac309a1aa9b8b7f14))
* **mcp-servers:** list only rows the caller can use ([#301](https://github.com/eve-esa/backend/issues/301)) ([9b8bed1](https://github.com/eve-esa/backend/commit/9b8bed1d2de0d8c839aef80689617356ea76fb6a))

## [1.0.3](https://github.com/eve-esa/backend/compare/v1.0.2...v1.0.3) (2026-10-06)


### Fixed

* **api-keys:** list keys from the primary right after a change ([#288](https://github.com/eve-esa/backend/issues/288)) ([1ad9182](https://github.com/eve-esa/backend/commit/1ad91829925b186f4da921dc965e7acdaaf58993))
* **documents:** index document ids on the private collection so deletes work ([#297](https://github.com/eve-esa/backend/issues/297)) ([e026767](https://github.com/eve-esa/backend/commit/e026767306c2b21ffaa8aee228bc8068409bc185))
* **llm:** warn about the missing Satcom models once per process ([#291](https://github.com/eve-esa/backend/issues/291)) ([77ac8fc](https://github.com/eve-esa/backend/commit/77ac8fca22b239d5df37777cd50aafba8fb5cbbf))
* **mcp:** shut the proxy sub-apps down without an error on rollover ([#292](https://github.com/eve-esa/backend/issues/292)) ([1e2160d](https://github.com/eve-esa/backend/commit/1e2160de77caebc85a7a5eeec710bf10bf6b0680))
* **messages:** only the owner can log a source click ([#296](https://github.com/eve-esa/backend/issues/296)) ([b707835](https://github.com/eve-esa/backend/commit/b7078356e6b458e8a07a3febf3f365f78fa66aa6))
* **observability:** export each span and log record once ([#298](https://github.com/eve-esa/backend/issues/298)) ([c047d20](https://github.com/eve-esa/backend/commit/c047d201f2734cafd4e8bf3dfe9cbd4b7f0400ce))
* **observability:** keep a user Stop out of the error spans ([#293](https://github.com/eve-esa/backend/issues/293)) ([0624586](https://github.com/eve-esa/backend/commit/062458655f681d71eedd7eaaa65f79a56172b490))
* **retrieval:** log a recovered embedding provider failure as a warning ([#295](https://github.com/eve-esa/backend/issues/295)) ([d11c580](https://github.com/eve-esa/backend/commit/d11c580970cadea8627b047d13ee77e64e3d2818))
* **retrieval:** search private collections without the year filter ([#294](https://github.com/eve-esa/backend/issues/294)) ([c0a8c14](https://github.com/eve-esa/backend/commit/c0a8c141a935477844284e9633b177e77edf1383))

## [1.0.2](https://github.com/eve-esa/backend/compare/v1.0.1...v1.0.2) (2026-10-05)


### Fixed

* **mcp:** stop ending sessions the stateless proxy cannot end ([#285](https://github.com/eve-esa/backend/issues/285)) ([73104fb](https://github.com/eve-esa/backend/commit/73104fb155ea0449e0e9d52c9f404d750b05c367))
* **stop:** persist a stop that lands before the first token ([#287](https://github.com/eve-esa/backend/issues/287)) ([720a880](https://github.com/eve-esa/backend/commit/720a88013c07099374be8944dbbb071284c83220))


### Changed

* **retrieval:** rerank with DeepInfra first, JSC as fallback ([#283](https://github.com/eve-esa/backend/issues/283)) ([3a4c6ff](https://github.com/eve-esa/backend/commit/3a4c6ffb765c56bcac6032fc97aecccb70166b39))
* **retrieval:** send rerankers trimmed candidates ([#286](https://github.com/eve-esa/backend/issues/286)) ([29261f3](https://github.com/eve-esa/backend/commit/29261f325179429ec683b0773eb5bd91ed318a37))

## [1.0.1](https://github.com/eve-esa/backend/compare/v1.0.0...v1.0.1) (2026-10-05)


### Fixed

* **db:** read the primary after a write when a check decides the answer ([#281](https://github.com/eve-esa/backend/issues/281)) ([2e19f7e](https://github.com/eve-esa/backend/commit/2e19f7e350e2afae694f815aa52b173b6a908edb))

## [1.0.0](https://github.com/eve-esa/backend/compare/v0.2.1...v1.0.0) (2026-10-05)

### Added
- Sign-in through the EVE identity provider; existing accounts keep their password and sign in once more.
- Streamed answers with Stop, keepalives on the stream, partial answer kept after a stop.
- OpenAI-compatible gateway at `/api/v1` with API keys and token budget charging (API key creation closed in prod, `API_KEY_MAX_ACTIVE_PER_USER=0`).
- Operator approval past the sign-up limit, with on hold, approved and welcome mails (`SIGNUP_AUTO_APPROVE_LIMIT`; prod 0 at launch, D16: every sign-up is approved, the gate can be re-armed with one tfvars change).
- Country and institution on the profile (flag FEATURE_PROFILE_FIELDS, prod off).
- Launch invite command for legacy accounts (`send_cohort_invite`, dry run by default).
- A retry notice when the service is busy: past the in-flight cap the backend answers 429 with `Retry-After`.
- Retention for conversation checkpoints (`CHECKPOINT_RETENTION_DAYS`, 30 days).
- Per-user request rate limit on chat, retrieval, uploads, `/api/v1` and MCP tool calls (flag FEATURE_REQUEST_RATE_LIMIT, prod off): 429 `rate_limited` with `Retry-After`, the chat retries once after the countdown.
- Behind flags: agentic chat (flag FEATURE_AGENTIC_CHAT, prod off), toolkits (flag FEATURE_TOOLKITS, prod off), model picker (flag FEATURE_MODEL_PICKER, prod off), custom models (flag FEATURE_CUSTOM_MODELS, prod off), API keys page (flag FEATURE_API_KEYS, prod off), attachments, artifacts, answered-by, bug report, status notices (prod off).
### Changed
- Answers come from an ordered model chain with failover, JSC first; token budgets apply to every chat route and to `/api/v1`.
- Retrieval no longer blocks the server: embeddings, rerank and Qdrant reads are async with a deadline and a provider fallback; the model breaker is shared across workers.
- Sources are grouped by document; the copy mark is sent only for persisted messages; the journal filter left the control panel.
- Deleting a collection that holds documents works on DocumentDB and frees the document slots.
- Copy without the pilot wording, base font follows the browser, About, Contact and Privacy links open their pages.
- Classification filters, My collections and the beta badge are flags (prod on).
### Removed
- Legacy auth endpoints and in-app login pages; `/users/me` no longer returns a password hash.
- `DELETE /artifacts/{id}`: an uploaded artifact is permanent.
- Dead settings, packages and modules (JWT settings, SATCOM Qdrant pair, SiliconFlow, fallback embedding key).

## [0.2.1](https://github.com/eve-esa/backend/compare/v0.2.0...v0.2.1) (2026-10-01)


### Fixed

* **hallucination:** judge follow-ups with the same history prefix a new turn sees ([#233](https://github.com/eve-esa/backend/issues/233)) ([b0c6220](https://github.com/eve-esa/backend/commit/b0c6220be99395ecc00fc129e0cb52a5b46164fb))

## [0.2.0](https://github.com/eve-esa/backend/compare/v0.1.5...v0.2.0) (2026-09-24)


### ⚠ BREAKING CHANGES

* **api-keys:** self-service key management with provenance, cascade revoke and a cap ([#220](https://github.com/eve-esa/backend/issues/220))
* **openai-gateway:** allowlist forwarded paths and charge the token budget ([#221](https://github.com/eve-esa/backend/issues/221))

### Added

* **agents:** record agentic setup time separately from total latency ([#227](https://github.com/eve-esa/backend/issues/227)) ([5a74376](https://github.com/eve-esa/backend/commit/5a74376e4116be15d9ea71dbeb93d76567e442eb))
* **api-keys:** self-service key management with provenance, cascade revoke and a cap ([#220](https://github.com/eve-esa/backend/issues/220)) ([c1c309b](https://github.com/eve-esa/backend/commit/c1c309b4469f62297133f17b1f3d3ac17c6c5246))
* **observability:** OpenTelemetry tracing and logs, Langfuse through the collector, bug reports ([#230](https://github.com/eve-esa/backend/issues/230)) ([9d7b6b9](https://github.com/eve-esa/backend/commit/9d7b6b96726211c939cdeea5fc1ea28ff86b2667))
* **openai-gateway:** allowlist forwarded paths and charge the token budget ([#221](https://github.com/eve-esa/backend/issues/221)) ([b7aba48](https://github.com/eve-esa/backend/commit/b7aba4881ed93c469f2e8fb325469bc019d430f0))


### Fixed

* **agents:** compute real per-step trace timings and fix tool content ([#225](https://github.com/eve-esa/backend/issues/225)) ([6ed58c7](https://github.com/eve-esa/backend/commit/6ed58c7e78d30558e2ed69e95b9b85796a1d3ab7))
* **agents:** keep retrieval chunk ids out of the model context ([#228](https://github.com/eve-esa/backend/issues/228)) ([ca504da](https://github.com/eve-esa/backend/commit/ca504da9ea324d0cdf2448980242b2a13cdd95e6))
* **agents:** trace streamed agent steps from the node update ([#226](https://github.com/eve-esa/backend/issues/226)) ([3f93f9b](https://github.com/eve-esa/backend/commit/3f93f9b9811592c1a6f89e0ef62afe5fb6b94b15))
* **openai-gateway:** resolve the route before reserving the token budget ([#223](https://github.com/eve-esa/backend/issues/223)) ([a5aa545](https://github.com/eve-esa/backend/commit/a5aa545a02f7fb8397e88487caec55a5d03bcffe))

## [0.1.5](https://github.com/eve-esa/backend/compare/v0.1.4...v0.1.5) (2026-09-17)


### Added

* **identity:** welcome email for every account that gets in ([#212](https://github.com/eve-esa/backend/issues/212)) ([3309f67](https://github.com/eve-esa/backend/commit/3309f67be651b81bf7e22a0bf4ec5f1a3474baf9))
* jsc embedders ([#214](https://github.com/eve-esa/backend/issues/214)) ([8e4a2b6](https://github.com/eve-esa/backend/commit/8e4a2b6957d8b65a619e1a535dec9ecd1896c602))
* **observability:** persist agentic failures to Mongo with a Langfuse overlay ([#217](https://github.com/eve-esa/backend/issues/217)) ([795695f](https://github.com/eve-esa/backend/commit/795695f1c78012490a068b89a944c3254f44efd8))


### Fixed

* **retrieval:** deduplicate before applying top-k ([#218](https://github.com/eve-esa/backend/issues/218)) ([a27e96e](https://github.com/eve-esa/backend/commit/a27e96e72e26836d5da3a4359715651d0b733a7a))
* **retrieval:** split Wiley envelope when MCP client pre-parses text to dict ([#216](https://github.com/eve-esa/backend/issues/216)) ([8415ea4](https://github.com/eve-esa/backend/commit/8415ea40d8ac7a419d3ce14c872b7c7356d65fe6))
* year based filtering for the documents during retrieval ([#215](https://github.com/eve-esa/backend/issues/215)) ([ee6992d](https://github.com/eve-esa/backend/commit/ee6992d75fbb2cfa8a7daaa7e08da491d5afe4e4))

## [0.1.4](https://github.com/eve-esa/backend/compare/v0.1.3...v0.1.4) (2026-09-09)


### Added

* **retrieval:** share public Qdrant collections across environments ([#207](https://github.com/eve-esa/backend/issues/207)) ([761b4c0](https://github.com/eve-esa/backend/commit/761b4c0f491f78af318c53bfec2d13edeb87ce21))

## [0.1.3](https://github.com/eve-esa/backend/compare/v0.1.2...v0.1.3) (2026-09-07)


### Added

* **identity:** on hold wording for the sign-up mails ([#205](https://github.com/eve-esa/backend/issues/205)) ([e43e251](https://github.com/eve-esa/backend/commit/e43e251a396814e70b30bf2c90717949e9d9ace2))

## [0.1.2](https://github.com/eve-esa/backend/compare/v0.1.1...v0.1.2) (2026-09-07)


### Added

* **identity:** soft limit on self sign-up with operator approval and emails ([#203](https://github.com/eve-esa/backend/issues/203)) ([f5252b9](https://github.com/eve-esa/backend/commit/f5252b9e11f7a401b941bc24c0d470805b6d9c1a))

## [0.1.1](https://github.com/eve-esa/backend/compare/v0.1.0...v0.1.1) (2026-09-03)


### Added

* **retrieval:** Qdrant multinenancy ([#199](https://github.com/eve-esa/backend/issues/199)) ([783e98b](https://github.com/eve-esa/backend/commit/783e98b9fc6a2301ef354b74ce51758f271601e5))


### Fixed

* **retrieval:** forward caller bearer as X-EVE-Token on MCP proxy egress ([#201](https://github.com/eve-esa/backend/issues/201)) ([6d32f97](https://github.com/eve-esa/backend/commit/6d32f9738cdeda5ff146dd807ecf107e85f9c656))

## [0.1.0](https://github.com/eve-esa/backend/compare/v0.0.11...v0.1.0) (2026-09-02)


### ⚠ BREAKING CHANGES

* the custom auth endpoints are gone; clients authenticate through the OIDC provider and send its access token as the bearer.

### Added

* add rewrite for better rag responses ([#192](https://github.com/eve-esa/backend/issues/192)) ([8e7473a](https://github.com/eve-esa/backend/commit/8e7473aaa2fb0b1e12bea7868876e79f30de83c7))
* authenticate users through the identity provider ([#196](https://github.com/eve-esa/backend/issues/196)) ([ea6fffc](https://github.com/eve-esa/backend/commit/ea6fffc31967df1e407403e0d07642f940b98b59))
* enforce the token budget on the agentic routes, and harden MCP server resolution ([#197](https://github.com/eve-esa/backend/issues/197)) ([cd4353f](https://github.com/eve-esa/backend/commit/cd4353f91920f23d2bf47e3226e40c84948af272))
* pass ui params to retrieve tool call ([#185](https://github.com/eve-esa/backend/issues/185)) ([9aa8868](https://github.com/eve-esa/backend/commit/9aa8868bb8febd129a7398ab2218b7945e45460b))


### Fixed

* **agentic:** make UI RAG params authoritative on retrieve tool calls ([#191](https://github.com/eve-esa/backend/issues/191)) ([22c1a7e](https://github.com/eve-esa/backend/commit/22c1a7e31c0256e35228c66e8e33241d89034d6d))

## [0.0.11](https://github.com/eve-esa/backend/compare/v0.0.10...v0.0.11) (2026-09-01)


### Added

* add rewrite for better rag responses ([#192](https://github.com/eve-esa/backend/issues/192)) ([8e7473a](https://github.com/eve-esa/backend/commit/8e7473aaa2fb0b1e12bea7868876e79f30de83c7))
* pass ui params to retrieve tool call ([#185](https://github.com/eve-esa/backend/issues/185)) ([9aa8868](https://github.com/eve-esa/backend/commit/9aa8868bb8febd129a7398ab2218b7945e45460b))


### Fixed

* **agentic:** make UI RAG params authoritative on retrieve tool calls ([#191](https://github.com/eve-esa/backend/issues/191)) ([22c1a7e](https://github.com/eve-esa/backend/commit/22c1a7e31c0256e35228c66e8e33241d89034d6d))
* hardcoded collections for staging ([#181](https://github.com/eve-esa/backend/issues/181)) ([7df23d9](https://github.com/eve-esa/backend/commit/7df23d901b223fd1d3c8b35868d8d4a60d742543))

## [0.0.10](https://github.com/eve-esa/backend/compare/v0.0.9...v0.0.10) (2026-08-13)


### Added

* encrypt custom-model keys with envelope encryption instead of per-key secrets ([#177](https://github.com/eve-esa/backend/issues/177)) ([6ac2515](https://github.com/eve-esa/backend/commit/6ac2515b6f72ef0a5dbe214f824fbb3240733d2c))
* the default EVE model answers via JSC where its picker entry is hidden ([#180](https://github.com/eve-esa/backend/issues/180)) ([8a3de20](https://github.com/eve-esa/backend/commit/8a3de20382551ccad1ba574d17798dd5d4bc483c))


### Fixed

* clear custom-model key material on delete ([#179](https://github.com/eve-esa/backend/issues/179)) ([d71f9cb](https://github.com/eve-esa/backend/commit/d71f9cb06df3ab2d2cd3cd2c9ff0082de0c654b2))

## [0.0.9](https://github.com/eve-esa/backend/compare/v0.0.8...v0.0.9) (2026-08-12)


### Added

* EVE answers through an ordered endpoint chain with failover ([#162](https://github.com/eve-esa/backend/issues/162)) ([4ef24d3](https://github.com/eve-esa/backend/commit/4ef24d3821269941a4839c3163a12c689c9a85b8))
* gate JSC platform model behind FEATURE_JSC_MODEL ([#159](https://github.com/eve-esa/backend/issues/159)) ([717afe7](https://github.com/eve-esa/backend/commit/717afe71b48a16a13f817adf85eed7cb15622454))
* structured tool fields on the agentic stream events ([#165](https://github.com/eve-esa/backend/issues/165)) ([428d933](https://github.com/eve-esa/backend/commit/428d9330156a7df02753c5e4237c294106227ce8))


### Fixed

* **deps:** clear the audited dependency vulnerabilities and automate updates ([#167](https://github.com/eve-esa/backend/issues/167)) ([13aeeec](https://github.com/eve-esa/backend/commit/13aeeecf614c376b26e7ae40fc3b7d83ed97d288))
* hold agentic generation until the stream subscriber attaches ([#164](https://github.com/eve-esa/backend/issues/164)) ([2d580e6](https://github.com/eve-esa/backend/commit/2d580e6fd6a232efc3686dabef04d72f4149c96a))
* inject the artifact instruction only on the first agentic turn ([#160](https://github.com/eve-esa/backend/issues/160)) ([d813156](https://github.com/eve-esa/backend/commit/d81315679e780bf02bacc95eb00d15eecd83ec25))
* open the agentic stream with a thinking notice ([#163](https://github.com/eve-esa/backend/issues/163)) ([e9fcbb2](https://github.com/eve-esa/backend/commit/e9fcbb2ed2e5f0c4681704daddd407126935264f))
* persist endpoint failures and keep retries on their pipeline ([#161](https://github.com/eve-esa/backend/issues/161)) ([fd1254f](https://github.com/eve-esa/backend/commit/fd1254fd8dbb53490d8266fe1dd2bf3d441b9f86))
* stop capping agentic answers with the endpoint probe budget ([#166](https://github.com/eve-esa/backend/issues/166)) ([dd0c7f6](https://github.com/eve-esa/backend/commit/dd0c7f69fcecfaffefe48177ee13644b51ece18b))
* unpack artifact_ids in message retry and keep SSE streams alive ([#157](https://github.com/eve-esa/backend/issues/157)) ([2065905](https://github.com/eve-esa/backend/commit/20659053f7ccd91e1e83263ded50d4b4bb6c37ad))

## [0.0.8](https://github.com/eve-esa/backend/compare/v0.0.7...v0.0.8) (2026-08-09)

### Added

* Feature flags are named for the features they control, each environment reports its own identity, and the OpenAI proxy no longer forwards the caller's token upstream ([#149](https://github.com/eve-esa/backend/pull/149))

### Removed

* `DELETE /artifacts/{id}`. An uploaded artifact is permanent ([#150](https://github.com/eve-esa/backend/pull/150))

### Fixed

* **ci:** report a missing dev image instead of dying on a raw AWS error ([#152](https://github.com/eve-esa/backend/issues/152)) ([207b0e4](https://github.com/eve-esa/backend/commit/207b0e44122f5a051434bc10e345e84cc202d105))
* **ci:** stop reporting success when a deploy started no tasks ([#153](https://github.com/eve-esa/backend/issues/153)) ([c27bf59](https://github.com/eve-esa/backend/commit/c27bf59b11c2227958ab36c38facbb779ae3bd62))

## [0.0.7] - 2026-08-07

### Added

- Agentic answers: `POST /conversations/{id}/generate-agentic` and `/stream-generate-agentic`. The
  assistant can plan and call external tools mid-conversation instead of answering in one shot, with a
  LangGraph react runner streaming tool calls and results over SSE.
- `/mcp-servers` CRUD, a catalog of MCP toolkits that are enabled or disabled individually and reached
  over HTTPS on Bedrock AgentCore with a bearer token.

### Changed

- `GET /mcp-servers/{id}` returns `tools_error`, so a toolkit that fails to answer is reported as
  failing rather than shown as an empty tool list. Previously "this server exposes nothing" and "we
  could not reach it" produced byte-identical responses, which made a wall of unreachable toolkits
  indistinguishable from a wall of empty ones.

Which toolkits exist is configuration per environment, not code. Production deliberately has no agentic
configuration.

## [0.0.6] - 2026-08-05

### Removed

- `sentence-transformers`, `transformers` and `langchain-huggingface` from `requirements.txt`. None
  of the three was ever imported: embeddings are computed by a remote OpenAI-compatible API through
  `EMBEDDING_URL`, not locally. They pulled in `torch`, which in its default PyPI form declares seven
  CUDA packages, on a service that runs on Fargate CPU and never sees a GPU.

  The image was 3.0 GB compressed, of which a single layer, the virtualenv, was 2,957 MB, and every
  Fargate task start downloads it from ECR before the process begins. It was paid on every deploy,
  every restart and every scale-out.

### Fixed

- URL-encode the account activation link. Plus-addressed emails (`user+tag@domain`) decoded to a space
  in the browser, so `/verify` answered "user not found". Affected both the activation and the
  resend-activation mails.

### Changed

- The ECS deploy step no longer keeps running for minutes after the service is already stable. The
  AWS SDK backs off exponentially up to 120 seconds between polls when left unconfigured; the
  interval is now pinned.
- Artifact storage: S3-backed user and MCP artifacts, a capture interceptor and the `/artifacts` API.
- Image rendering support: demo image catalog, answer-side artifact stubs, `artifact_ids` over SSE, and
  catalog image URLs rewritten at the `generate_answer` boundary.
- User attachments are persisted on the agentic endpoints.

## [0.0.5] - 2026-08-02

### Added

- The running version and commit are reported at `GET /api/health` and set on the FastAPI
  application, so an environment can be asked what it is serving instead of being assumed.

First version that reports itself. Before this, no environment could be asked which build it was
running.

## [0.0.4] - 2026-07-31

Promotion drill with the environment source read inside the gated job.

## [0.0.3] - 2026-07-31

Single-approval promotion, and resilient index creation on a fresh DocumentDB.

## [0.0.2] - 2026-07-30

Promotion drill with the corrected pipeline: version tag normalisation and conditional health check.

## [0.0.1] - 2026-07-30

First promotion drill: validates tag-gated staging promotion by image digest.
