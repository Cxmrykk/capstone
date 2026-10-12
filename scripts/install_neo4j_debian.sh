#!/usr/bin/env bash
#
# Bare-metal Neo4j on Debian 13 (trixie). No Docker.
#
# Neo4j Community Edition serves exactly ONE database. You can only evaluate
# against one dataset alias at a time locally -- which is why the default
# execution provider is `demo` (Neo4j Labs' hosted server). Use this local
# install for development, offline work, or when the demo server is down.
#
# Usage:  bash scripts/install_neo4j_debian.sh
set -euo pipefail

BOLD=$'\033[1m'; RESET=$'\033[0m'; YELLOW=$'\033[33m'
say() { echo "${BOLD}==>${RESET} $*"; }
warn() { echo "${YELLOW}!!${RESET} $*"; }

if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

say "Installing prerequisites (curl, gnupg, Java 21)"
$SUDO apt-get update
$SUDO apt-get install -y curl gnupg ca-certificates apt-transport-https openjdk-21-jre-headless

say "Adding the Neo4j APT repository"
$SUDO install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key \
    | $SUDO gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg
$SUDO chmod 0644 /etc/apt/keyrings/neotechnology.gpg

echo "deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest" \
    | $SUDO tee /etc/apt/sources.list.d/neo4j.list > /dev/null

say "Installing Neo4j Community Edition"
$SUDO apt-get update
$SUDO apt-get install -y neo4j

CONF=/etc/neo4j/neo4j.conf
say "Tuning ${CONF} for standard workstations (e.g. 16 GB RAM baseline)"
$SUDO cp "${CONF}" "${CONF}.bak.$(date +%s)"

set_conf() {
    local key="$1" value="$2"
    if $SUDO grep -qE "^#?\s*${key}=" "${CONF}"; then
        $SUDO sed -i -E "s|^#?\s*${key}=.*|${key}=${value}|" "${CONF}"
    else
        echo "${key}=${value}" | $SUDO tee -a "${CONF}" > /dev/null
    fi
}

# Conservative heap/page-cache so Neo4j does not fight the host applications for RAM.
set_conf "server.memory.heap.initial_size" "1G"
set_conf "server.memory.heap.max_size" "2G"
set_conf "server.memory.pagecache.size" "1G"
set_conf "server.default_listen_address" "127.0.0.1"
# Read-only: the evaluation harness never needs to write, and this removes any
# chance of a generated query mutating the graph.
set_conf "dbms.databases.default_to_read_only" "true"

say "Enabling and starting the service"
$SUDO systemctl enable neo4j
$SUDO systemctl restart neo4j

say "Waiting for Neo4j to accept connections"
for _ in $(seq 1 30); do
    if curl -sf http://localhost:7474 > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

cat <<EOF

$(say "Done")

Next steps
----------
1. Set the initial password (default credentials are neo4j/neo4j):

     ${BOLD}sudo neo4j-admin dbms set-initial-password 'CHOOSE_A_PASSWORD'${RESET}
     ${BOLD}sudo systemctl restart neo4j${RESET}

   Then export it so the evaluation harness can authenticate:

     ${BOLD}export NEO4J_PASSWORD='CHOOSE_A_PASSWORD'${RESET}

2. Load a demo dump (one at a time -- Community Edition hosts one database):

     ${BOLD}sudo systemctl stop neo4j${RESET}
     ${BOLD}sudo -u neo4j neo4j-admin database load neo4j --from-path=/path/to/dumps --overwrite-destination=true${RESET}
     ${BOLD}sudo systemctl start neo4j${RESET}

3. Point the harness at it:

     ${BOLD}python app.py neo4j check --provider local --alias neo4jlabs_demo_db_movies${RESET}

   and in your config set:

     neo4j:
       provider: local
       local_alias: neo4jlabs_demo_db_movies

4. Browser UI: http://localhost:7474

$(warn "For full-coverage execution accuracy across all 20+ dataset databases, keep provider: demo -- that hits Neo4j Labs' hosted server and needs no local dumps.")
EOF
