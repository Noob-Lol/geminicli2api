import logging
import platform

from .config import CLI_VERSION

logger = logging.getLogger("geminicli2api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_user_agent():
    """Generate User-Agent string matching gemini-cli format."""
    version = CLI_VERSION
    system = platform.system()
    arch = platform.machine()
    return f"GeminiCLI/{version} ({system}; {arch})"


def get_platform_string():
    """Generate platform string matching gemini-cli format."""
    system = platform.system().upper()
    arch = platform.machine().upper()

    # Map to gemini-cli platform format
    if system == "DARWIN":
        if arch in {"ARM64", "AARCH64"}:
            return "DARWIN_ARM64"
        return "DARWIN_AMD64"
    if system == "LINUX":
        if arch in {"ARM64", "AARCH64"}:
            return "LINUX_ARM64"
        return "LINUX_AMD64"
    if system == "WINDOWS":
        return "WINDOWS_AMD64"
    return "PLATFORM_UNSPECIFIED"


def get_client_metadata(project_id=None):
    return {
        "ideType": "IDE_UNSPECIFIED",
        "platform": get_platform_string(),
        "pluginType": "GEMINI",
        "duetProject": project_id,
    }
