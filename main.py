import os
import sys
from dotenv import load_dotenv
from managers.pipeline_factory import PipelineFactory
from agents.orchestrator import OrchestratorAgent

load_dotenv()

_REQUIRED = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def _check_env() -> None:
    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        print("Add them to .env and try again.")
        sys.exit(1)


def main() -> None:
    _check_env()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run main.py script.txt")
        print('  uv run main.py "Your script text here..."')
        sys.exit(1)

    arg = sys.argv[1]
    script = open(arg, encoding="utf-8").read() if os.path.isfile(arg) else arg

    print(f"Script loaded ({len(script)} chars)\n")

    pipeline = PipelineFactory.production()
    output_path = OrchestratorAgent(pipeline).run(script)

    print(f"\n{'='*50}")
    print(f"Done!  Video saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
