"""
Tests for deduplication module.
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from src.data.dedup import (
    compute_sha256,
    compute_phash,
    hamming_distance,
    find_exact_duplicates
)


@pytest.fixture
def temp_images():
    """Create temporary test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a base image
        img1 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img1_path = tmpdir / "img1.png"
        img1.save(img1_path)
        
        # Create exact duplicate
        img2_path = tmpdir / "img2.png"
        img1.save(img2_path)
        
        # Create different image
        img3 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img3_path = tmpdir / "img3.png"
        img3.save(img3_path)
        
        yield tmpdir, [img1_path, img2_path, img3_path]


def test_compute_sha256(temp_images):
    """Test SHA256 computation."""
    tmpdir, paths = temp_images
    
    hash1 = compute_sha256(paths[0])
    hash2 = compute_sha256(paths[1])  # Exact copy
    hash3 = compute_sha256(paths[2])  # Different image
    
    assert hash1 == hash2, "Exact copies should have same hash"
    assert hash1 != hash3, "Different images should have different hashes"
    assert len(hash1) == 64, "SHA256 should be 64 hex characters"


def test_compute_phash(temp_images):
    """Test perceptual hash computation."""
    tmpdir, paths = temp_images
    
    phash1 = compute_phash(paths[0])
    phash2 = compute_phash(paths[1])
    
    assert phash1 is not None
    assert phash1 == phash2, "Exact copies should have same phash"


def test_hamming_distance():
    """Test Hamming distance calculation."""
    # Same hash
    dist = hamming_distance("a" * 16, "a" * 16)
    assert dist == 0
    
    # None handling
    dist = hamming_distance(None, "a" * 16)
    assert dist == float("inf")


def test_find_exact_duplicates(temp_images):
    """Test exact duplicate detection."""
    tmpdir, paths = temp_images
    
    duplicates = find_exact_duplicates(paths)
    
    # Should find one set of duplicates (img1 and img2)
    assert len(duplicates) == 1
    
    # The duplicate set should contain 2 paths
    dup_set = list(duplicates.values())[0]
    assert len(dup_set) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
