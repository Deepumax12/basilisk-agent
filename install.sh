#!/usr/bin/env bash
# Install Basilisk Agent for the current user: links `basilisk-agent` into ~/.local/bin.
# Run it from the cloned repository:   ./install.sh
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    echo "Python 3.8 or newer is needed (sudo apt install python3)." >&2
    exit 1
fi

chmod +x "$DIR/basilisk_agent.py"
mkdir -p "$HOME/.local/bin" "$HOME/.basilisk_agent"
ln -sf "$DIR/basilisk_agent.py" "$HOME/.local/bin/basilisk-agent"
echo "Linked: ~/.local/bin/basilisk-agent -> $DIR/basilisk_agent.py"

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *)
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        echo "Added ~/.local/bin to your PATH in ~/.bashrc. Open a new terminal (or run: source ~/.bashrc)."
        ;;
esac

if [ ! -f "$HOME/.basilisk_agent/config.json" ]; then
    cp "$DIR/config.example.json" "$HOME/.basilisk_agent/config.json"
    echo "Created ~/.basilisk_agent/config.json (edit it to choose your AI and model)."
fi

missing=""
for m in numpy matplotlib PIL; do
    python3 -c "import $m" 2>/dev/null || missing="$missing $m"
done
if [ -n "$missing" ]; then
    echo
    echo "Optional Python packages missing:$missing"
    echo "  sudo apt install python3-numpy python3-matplotlib python3-pil"
fi

echo
echo "Done. Next steps:"
echo "  basilisk-agent sim env      # check your machine (no AI needed)"
echo "  basilisk-agent --check      # check the AI connection (Claude or Ollama)"
echo "Update later with:  cd \"$DIR\" && git pull"
