"""
Kernel loader for flash attention implementations.
Downloads and loads flash attention kernels from GitHub repositories.

Usage:
    from kernels import get_kernel
    
    # Get kernel interface
    kernel_module = get_kernel("kernels-community/flash-attn3")
    fa_interface = kernel_module.flash_attn_interface
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def get_kernel(repo: str):
    """
    Load a flash attention kernel from the specified GitHub repository.
    
    Args:
        repo: GitHub repository path (e.g., "kernels-community/flash-attn3")
        
    Returns:
        Module with flash_attn_interface function
    """
    # Check environment variable for custom path
    custom_path = os.environ.get("KERNELS_PATH")
    if custom_path and os.path.exists(custom_path):
        spec = importlib.util.spec_from_file_location("kernels", custom_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Determine cache directory
    cache_dir = Path.home() / ".cache" / "autoresearch" / "kernels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    repo_name = repo.replace("/", "-")
    repo_cache_dir = cache_dir / repo_name
    
    # Try to import from cached location first
    try:
        spec = importlib.util.spec_from_file_location(
            "flash_attn", 
            str(repo_cache_dir / "flash_attn.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    
    # Check if package is already installed
    try:
        if repo == "varunneal/flash-attention-3":
            from flash_attn import flash_attn_func as fa_interface
        else:
            from flash_attn import flash_attn_func as fa_interface
        
        class Module:
            flash_attn_interface = staticmethod(fa_interface)
        
        return Module()
    except ImportError:
        pass
    
    # Clone repository if not present
    print(f"Downloading kernel from {repo}...")
    repo_cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git"],
            cwd=cache_dir,
            check=True,
            capture_output=True
        )
        
        # Find the actual cloned directory name
        clone_name = repo_name.replace("flash-attn3", "").rstrip("-") or repo_name
        clone_path = cache_dir / f"{repo}.git"
        
        if not clone_path.exists():
            # Try alternate naming
            for item in cache_dir.iterdir():
                if item.is_dir() and "flash" in str(item).lower():
                    clone_path = item
                    break
        
        if clone_path.exists():
            # Copy the flash_attn.py file to our cache location
            src_file = clone_path / "flash_attn.py"
            if not src_file.exists():
                # Try common alternative locations
                alt_files = [
                    clone_path / "csrc" / "flash_attn.cpp",
                    clone_path / "flash_attention.cu",
                ]
                for alt in alt_files:
                    if alt.exists():
                        src_file = alt
                        break
            
            # If we found the source, create a wrapper module
            if src_file.exists():
                _create_kernel_wrapper(repo_cache_dir, clone_path)
            
            # Try to import again
            try:
                spec = importlib.util.spec_from_file_location(
                    "flash_attn", 
                    str(repo_cache_dir / "flash_attn.py")
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            except Exception as e:
                print(f"Warning: Could not load kernel from cloned repo: {e}")
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to clone repository: {e}")
    
    # Fallback: use PyTorch's scaled_dot_product_attention if available
    print("Fallback: Using torch.nn.functional.scaled_dot_product_attention")
    return _create_fallback_kernel()


def _create_kernel_wrapper(cache_dir, repo_path):
    """Create a Python wrapper for the flash attention implementation."""
    
    # Create a simple wrapper that uses the installed package
    wrapper_code = '''"""
Flash Attention interface wrapper.
Auto-generated - uses underlying installation.
"""

try:
    from flash_attn import flash_attn_func as fa_interface
except ImportError:
    try:
        from flash_attn_qkvpacked_func import flash_attn_qkvpacked_func as fa_interface
    except ImportError:
        # Fallback to torch's SDPA
        import torch.nn.functional as F
        
        def _sdpa_fallback(q, k, v, causal=False, window_size=(-1, -1)):
            """Use scaled_dot_product_attention as fallback."""
            scale = q.size(-1) ** -0.5
            if window_size[0] >= 0 or window_size[1] >= 0:
                # Create causal mask with window
                b, t, _, _ = q.shape
                # Simple sliding window attention
                attn = torch.matmul(q * scale, k.transpose(-2, -1))
                # Apply causal and window masks
                mask = torch.full((t, t), float("-inf"), device=q.device)
                for i in range(t):
                    start = max(0, i - window_size[0]) if window_size[0] >= 0 else 0
                    end = min(t, i + window_size[1] + 1) if window_size[1] >= 0 else t
                    mask[i, start:end] = 0
                attn = attn + mask
                attn = torch.softmax(attn, dim=-1)
                y = torch.matmul(attn, v)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            return y
        
        fa_interface = _sdpa_fallback

flash_attn_interface = fa_interface
'''
    
    wrapper_path = cache_dir / "flash_attn.py"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_code)


def _create_fallback_kernel():
    """Create a fallback kernel using torch's scaled_dot_product_attention."""
    import torch.nn.functional as F
    
    def _sdpa_fallback(q, k, v, causal=False, window_size=(-1, -1)):
        """Use scaled_dot_product_attention as fallback."""
        scale = q.size(-1) ** -0.5
        
        if window_size[0] >= 0 or window_size[1] >= 0:
            # Create sliding window attention
            b, t, _, _ = q.shape
            attn = torch.matmul(q * scale, k.transpose(-2, -1))
            
            mask = torch.full((t, t), float("-inf"), device=q.device)
            for i in range(t):
                start = max(0, i - window_size[0]) if window_size[0] >= 0 else 0
                end = min(t, i + window_size[1] + 1) if window_size[1] >= 0 else t
                mask[i, start:end] = 0
            
            attn = attn + mask
            attn = torch.softmax(attn, dim=-1)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        
        return y
    
    class Module:
        flash_attn_interface = staticmethod(_sdpa_fallback)
    
    return Module()


if __name__ == "__main__":
    # Test the kernel loading
    print("Testing kernel loading...")
    
    for repo in ["varunneal/flash-attention-3", "kernels-community/flash-attn3"]:
        try:
            module = get_kernel(repo)
            print(f"Successfully loaded kernel from {repo}")
        except Exception as e:
            print(f"Failed to load kernel from {repo}: {e}")