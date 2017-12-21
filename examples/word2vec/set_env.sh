#!/usr/bin/env zsh

here=$(dirname $(readlink -f $0))
export PYTHONPATH=$here/python_module:$PYTHONPATH
here=""

# mkdir -p other_modules 1>/dev/null 2>/dev/null
# pushd other_modules
# for repo_name in 'text_processing'; do
#     if [ -e "$repo_name" ]; then
#         pushd $repo_name
#         git pull
#         popd
#     else
#         git clone "https://github.com/HonoMi/"$repo_name
#     fi
#     source $module_name/set_env.sh
# done
# popd
