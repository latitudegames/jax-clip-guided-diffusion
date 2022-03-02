#!/bin/bash
# This is a dirty hack to let us pass arbitrary arguments to model.py as a single string.

# The user of this file will call it like this
# CMD ["main.sh", "{ \"prompt\": \"Big Al's best friend\", \"iterations\": 300}"]
# Where the only argument to this script ($1) is a json string with key-value argument value pairs.

# The first python script will turn this into
# --prompt
# Big Al's best friend
# --iterations
# 300

# And xarg will glue those together into a single command
printf "Received input:\n"
printf "%s\n" "$1"
python json_to_bash_args.py "$1" | xargs -0 -d "\n" python model.py