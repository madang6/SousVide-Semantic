#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/admin/StanfordMSL/SousVide-Semantic/configs/experiment/ssv_multi3dgs.yml"
TMP="/home/admin/StanfordMSL/SousVide-Semantic/configs/experiment/ssv_multi3dgs.tmp.yml"

# reverse order
FLIGHTS=( "flightroom_lowres" "sv_917_3_left_gemsplat" "spheres" "indooroutdoor" "packardpark")
# FLIGHTS=( "sv_917_3_left_gemsplat" "spheres" "indooroutdoor" "packardpark")
# FLIGHTS=( "spheres" "indooroutdoor" "packardpark")
# FLIGHTS = ( "spheres" ) 
# FLIGHTS=("flightroom_lowres") 

for flight in "${FLIGHTS[@]}"; do
  echo "=== Running flight = $flight ==="

  perl -pe '
    if (/^[[:space:]]*-\s*\[/)      { s/^/#/; }
    if (/^([[:space:]]*)#\s*-\s*\[.*'"$flight"'.*\]/) { s/^([[:space:]]*)#\s*/\1/; }
  ' "$CONFIG" > "$TMP"

  # build the args in an array
  args=( generate-rollouts --config-file "$TMP" )
  
  # args+=( --validation-mode )

  args+=( --use-wandb --wandb-project ssv )
  # args+=( --use-wandb --wandb-project ssv \
  #         --wandb-run-id clbyewvn --wandb-resume allow )

  python ssv_multi3dgs_campaign.py "${args[@]}"
done

rm -f "$TMP"

# #!/usr/bin/env bash
# set -euo pipefail

# CONFIG="/home/admin/StanfordMSL/SousVide-Semantic/configs/experiment/ssv_multi3dgs.yml"
# TMP="/home/admin/StanfordMSL/SousVide-Semantic/configs/experiment/ssv_multi3dgs.tmp.yml"

# # flights in reverse order
# FLIGHTS=(
#   "indooroutdoor"
#   "spheres"
#   "flightroom_lowres"
#   "sv_917_3_left_gemsplat"
# )

# for flight in "${FLIGHTS[@]}"; do
#   echo
#   echo "=== DRY RUN: processing flight = $flight ==="
#   perl -pe '
#     # 1) comment every “- [ … ]” line
#     if (/^[[:space:]]*-\s*\[.*\]/) {
#       s/^/#/;
#     }
#     # 2) if this is the one we want, strip the "#"
#     if (/^([[:space:]]*)#\s*-\s*\[.*'"$flight"'.*\]/) {
#       # remove the "#" but preserve leading spaces from $1
#       s/^([[:space:]]*)#\s*/\1/;
#     }
#   ' "$CONFIG" > "$TMP"

#   echo "--- head of $TMP ---"
#   head -n 20 "$TMP"
#   echo "--------------------"

#   echo
#   echo "--- would invoke: ---"
#   echo python ssv_multi3dgs_campaign.py generate-rollouts \
#        --config-file "$TMP" \
#        --validation-mode \
#        --use-wandb --wandb-project ssv \
#        --wandb-run-id 6uzo8pgf \
#        --wandb-resume allow
#   echo "---------------------"

#   break
# done

# rm -f "$TMP"
