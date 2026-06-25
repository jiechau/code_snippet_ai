

tutorial:
https://microsoft.github.io/graphrag/posts/get_started/

mkdir -p ./ragtest_ch/input
cp xxx ./ragtest_ch/input
python -m graphrag.index --init --root ./ragtest_ch
vi ragtest_ch/.env
vi ragtest_ch/settings.yaml # model: gpt-4o-mini
python -m graphrag.index --root ./ragtest_ch
python -m graphrag.query --root ./ragtest_ch --method global "這本書的主旨是什麼?"
python -m graphrag.query --root ./ragtest_ch --method global "這篇論文主要在說明什麼?"
python -m graphrag.query --root ~/ragtest_en --method global "what is main idea of this book?"


