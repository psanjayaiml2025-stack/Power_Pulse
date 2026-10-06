from fastapi import APIRouter
from ..ml.official_benchmark import get_official_benchmark

router = APIRouter(prefix="/api/benchmark", tags=["official-benchmark"])

@router.get("")
def benchmark():
    return get_official_benchmark()
