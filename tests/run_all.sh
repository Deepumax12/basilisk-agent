#!/usr/bin/env bash
# Run all tests. They use mock AI servers and a stand-in solver, so they need
# no API key, no Ollama and no Basilisk install (gcc, numpy, matplotlib and Pillow are needed).
cd "$(dirname "${BASH_SOURCE[0]}")"
status=0
for t in test_tools.py test_loop.py test_interactive.py test_local.py test_sim_cli.py; do
    printf "%-22s " "$t"
    if timeout 900 python3 "$t" > "/tmp/basilisk-agent-$t.log" 2>&1; then
        echo "ok   ($(tail -1 "/tmp/basilisk-agent-$t.log"))"
    else
        echo "FAILED (see /tmp/basilisk-agent-$t.log)"
        status=1
    fi
done
exit $status
