from .apwg import iter_apwg
from .duplicates import duplicate_groups, without_test_duplicates
from .manifest import read_manifest, slug, write_manifest
from .phishpedia import iter_phishpedia
from .splits import (
    BenchmarkSpec,
    build_benchmark,
    credential_login_pages,
    is_lookalike,
    stratified_split,
)
from .trop import iter_trop

__all__ = [
    "BenchmarkSpec",
    "build_benchmark",
    "credential_login_pages",
    "duplicate_groups",
    "iter_apwg",
    "iter_phishpedia",
    "is_lookalike",
    "iter_trop",
    "read_manifest",
    "slug",
    "stratified_split",
    "without_test_duplicates",
    "write_manifest",
]
