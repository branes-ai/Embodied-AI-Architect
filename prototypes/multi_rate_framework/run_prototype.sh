#!/bin/bash

# Script to run the multi-rate framework prototype

echo "=================================================="
echo "Multi-Rate Control Framework Prototype"
echo "=================================================="
echo ""

# Check if zenoh is installed
python3 -c "import zenoh" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Zenoh not installed"
    echo ""
    echo "Installing eclipse-zenoh..."
    pip install eclipse-zenoh
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Zenoh"
        echo "Try: pip install eclipse-zenoh"
        exit 1
    fi
    echo "✅ Zenoh installed"
    echo ""
fi

# Menu
echo "Select example to run:"
echo "  1) Simple (100 Hz producer, 1 Hz consumer)"
echo "  2) Multi-rate (100 Hz sensor, 10 Hz control, 1 Hz planning)"
echo "  3) Run both sequentially"
echo ""
read -p "Choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Running simple example..."
        echo ""
        python3 example_simple.py
        ;;
    2)
        echo ""
        echo "Running multi-rate example..."
        echo ""
        python3 example_multirate.py
        ;;
    3)
        echo ""
        echo "Running simple example (Ctrl+C to stop and continue)..."
        echo ""
        python3 example_simple.py

        echo ""
        echo ""
        echo "Running multi-rate example (Ctrl+C to stop)..."
        echo ""
        python3 example_multirate.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
