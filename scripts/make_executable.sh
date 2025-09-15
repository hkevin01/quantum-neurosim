#!/bin/bash
# Make all scripts executable

chmod +x scripts/start.sh
chmod +x scripts/stop.sh
chmod +x scripts/backup.sh

echo "✅ All scripts are now executable!"
echo ""
echo "🚀 Ready to start your quantum development environment:"
echo "   ./scripts/start.sh                    # Basic CPU environment"
echo "   ./scripts/start.sh --gpu              # Include GPU acceleration"
echo "   ./scripts/start.sh --monitoring       # Full monitoring stack"
echo ""
echo "📚 Open the tutorial at: notebooks/quantum_docker_tutorial.ipynb"
