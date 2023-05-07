cd pytests/
coverage run --omit="*/OpenNE*" -m pytest
coverage html --omit="*/test*,*/OpenNE*"
mkdir -p Coverage
rm Coverage/coverage.svg
coverage-badge -o Coverage/coverage.svg