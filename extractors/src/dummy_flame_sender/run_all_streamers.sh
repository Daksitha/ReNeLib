title1="MFCC"
title2="Listener_FLAME"
title3="Speaker_FLAME"

cmd1="python MFCC_Speaker_Audio.py"
cmd2="python FLAME_listener_streamer.py"
cmd3="python FLAME_speaker_streamer.py"

gnome-terminal --tab --title="$title1" --command="bash -c '$cmd1; $SHELL'" \
               --tab --title="$title2" --command="bash -c '$cmd2; $SHELL'" \
               --tab --title="$title3" --command="bash -c '$cmd3; $SHELL'"
