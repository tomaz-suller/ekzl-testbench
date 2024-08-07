rsync \
    --archive \
    --cvs-exclude \
    --human-readable \
    --partial \
    --recursive \
    --update \
    --verbose \
    --exclude-from='.gitignore' \
    . \
    se2:~/ezkl
