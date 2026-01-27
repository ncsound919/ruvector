#!/bin/bash

# RuvBot Installation Script
# Usage: curl -fsSL https://get.ruvector.dev/ruvbot | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "  ____             ____        _   "
echo " |  _ \ _   ___   _| __ )  ___ | |_ "
echo " | |_) | | | \ \ / /  _ \ / _ \| __|"
echo " |  _ <| |_| |\ V /| |_) | (_) | |_ "
echo " |_| \_\\\\__,_| \_/ |____/ \___/ \__|"
echo -e "${NC}"
echo "Self-learning AI Assistant"
echo ""

# Check Node.js version
check_node() {
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js is not installed.${NC}"
        echo "Please install Node.js 18 or later: https://nodejs.org"
        exit 1
    fi

    NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        echo -e "${RED}Error: Node.js 18+ required. Found v$(node -v)${NC}"
        exit 1
    fi

    echo -e "${GREEN}[x] Node.js $(node -v)${NC}"
}

# Check npm version
check_npm() {
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}Error: npm is not installed.${NC}"
        exit 1
    fi

    echo -e "${GREEN}[x] npm $(npm -v)${NC}"
}

# Install RuvBot globally
install_ruvbot() {
    echo ""
    echo -e "${BLUE}Installing @ruvector/ruvbot...${NC}"

    npm install -g @ruvector/ruvbot

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[x] RuvBot installed successfully!${NC}"
    else
        echo -e "${RED}Installation failed. Try running with sudo:${NC}"
        echo "sudo npm install -g @ruvector/ruvbot"
        exit 1
    fi
}

# Show next steps
show_next_steps() {
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Initialize a new project:"
    echo -e "     ${YELLOW}ruvbot init${NC}"
    echo ""
    echo "  2. Configure your environment:"
    echo -e "     ${YELLOW}ruvbot config${NC}"
    echo ""
    echo "  3. Start the bot:"
    echo -e "     ${YELLOW}ruvbot start${NC}"
    echo ""
    echo "Documentation: https://github.com/ruvnet/ruvector"
    echo ""
}

# Main installation flow
main() {
    echo "Checking requirements..."
    echo ""

    check_node
    check_npm

    install_ruvbot
    show_next_steps
}

main
