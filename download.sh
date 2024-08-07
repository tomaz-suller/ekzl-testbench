rsync \
    --archive \
    --cvs-exclude \
    --human-readable \
    --partial \
    --recursive \
    --update \
    --info=PROGRESS2 \
    se2:~/ezkl/data \
    se2:~/ezkl/models \
    se2:~/ezkl/multirun \
    se2:~/ezkl/output \
    se2:~/ezkl/outputs \
    .
