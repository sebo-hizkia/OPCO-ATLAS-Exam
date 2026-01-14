import time
import uuid
from fastapi import Request
from loguru import logger


async def audit_requests(request: Request, call_next):
    # Génération d’un ID de session (ou récupération future)
    session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))

    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        "REQUEST | session={session} | method={method} | path={path} | "
        "status={status} | duration={duration:.2f}ms",
        session=session_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration=duration_ms
    )

    return response
