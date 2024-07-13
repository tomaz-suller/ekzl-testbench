rsync \
    --archive \
    --cvs-exclude \
    --human-readable \
    --partial \
    --recursive \
    --update \
    --info=PROGRESS2 \
    se2:~/ezkl/output \
    se2:~/ezkl/outputs \
    se2:~/ezkl/multirun \
    .

