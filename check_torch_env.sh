#!/usr/bin/env bash
echo "======== 환경 정보 ========" 
echo "OS 정보:" 
uname -a 
echo 
echo "Python 버전:" 
python --version 2>&1 
echo 
echo "pip 버전:" 
pip --version 2>&1 
echo 
echo "conda 환경:" 
conda info --envs | grep '^\*' 
echo 

echo "======== 설치된 torch 관련 ========" 
echo "site-packages/torch 디렉토리 구조:" 
python - <<PYCODE
import site, os
paths = site.getsitepackages() + [site.getusersitepackages()]
for p in paths:
    tdir = os.path.join(p, 'torch')
    if os.path.isdir(tdir):
        print("->", tdir)
        for root, dirs, files in os.walk(tdir):
            for name in files:
                print(os.path.relpath(os.path.join(root, name), p))
PYCODE
echo 

echo "======== import torch 테스트 ========" 
python - <<'EOF2'
try:
    import torch
    print("✅ torch import 성공:", torch.__version__)
except Exception as e:
    print(f"❌ import torch 실패: {e!r}")
EOF2
