### 1. CHOOSE SCALE ###
SCALE=4
# SCALE=6
# SCALE=8
# SCALE=A
# SCALE=C

### 2. CHOOSE METHOD ###
METHOD=A
# METHOD=E
# METHOD=R
# METHOD=Z

### 3. PROCESS ###
if [[ ${SCALE} = '4' ]]; then
  SCALE_STR=16M-13p1B-100k-bs64-lr
elif [[ ${SCALE} = '6' ]]; then
  SCALE_STR=29M-13p1B-100k-bs64-lr
elif [[ ${SCALE} = '8' ]]; then
  SCALE_STR=57M-13p1B-100k-bs64-lr
elif [[ ${SCALE} = 'A' ]]; then
  SCALE_STR=109M-13p1B-100k-bs64-lr
elif [[ ${SCALE} = 'C' ]]; then
  SCALE_STR=221M-13p1B-100k-bs64-lr
fi

if [[ ${METHOD} = 'A' ]]; then
  METHOD_STR=baseline-g0e+00
elif [[ ${METHOD} = 'E' ]]; then
  METHOD_STR=muloss-g1e-04
elif [[ ${METHOD} = 'R' ]]; then
  METHOD_STR=mucentering-g0e+00
elif [[ ${METHOD} = 'Z' ]]; then
  METHOD_STR=zloss-g1e-04
fi

### 4. EXECUTE ###
cd nanoGPT

torchrun --standalone --nproc_per_node=4 train.py config/wor${SCALE}0${METHOD}-${SCALE_STR}3-${METHOD_STR}-s1.py
# torchrun --standalone --nproc_per_node=4 train.py config/wor${SCALE}0${METHOD}-${SCALE_STR}10-${METHOD_STR}-s1.py
# torchrun --standalone --nproc_per_node=4 train.py config/wor${SCALE}0${METHOD}-${SCALE_STR}30-${METHOD_STR}-s1.py
# torchrun --standalone --nproc_per_node=4 train.py config/wor${SCALE}0${METHOD}-${SCALE_STR}100-${METHOD_STR}-s1.py
# torchrun --standalone --nproc_per_node=4 train.py config/wor${SCALE}0${METHOD}-${SCALE_STR}300-${METHOD_STR}-s1.py
# torchrun --standalone --nproc_per_node=4 train.py config/wor${SCALE}0${METHOD}-${SCALE_STR}1000-${METHOD_STR}-s1.py
# torchrun --standalone --nproc_per_node=4 train.py config/wor${SCALE}0${METHOD}-${SCALE_STR}3000-${METHOD_STR}-s1.py
