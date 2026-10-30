#!/bin/sh
# Download the third-party resources the lyric normalizer reads at runtime and check them against the
# hashes the released labels were built with. A mismatch means upstream published a newer file: labels
# (Japanese readings, language identification) may then differ from the ones the models were trained on.
#   sh scripts/fetch_resources.sh            # into src/echora_kara/text/dictionaries and resources/
set -eu
cd "$(dirname "$0")/.."
dict=src/echora_kara/text/dictionaries
mkdir -p "$dict" resources

fetch() {  # url destination
  [ -s "$2" ] && { echo "have $2"; return; }
  echo "fetching $2"
  curl -fsSL -o "$2.part" "$1" && mv "$2.part" "$2"
}

# JMdict furigana segmentation (CC BY-SA 4.0, EDRDG licence) and KANJIDIC2 (CC BY-SA 4.0, EDRDG licence).
fetch https://github.com/Doublevil/JmdictFurigana/releases/latest/download/JmdictFurigana.txt "$dict/JmdictFurigana.txt"
fetch http://www.edrdg.org/kanjidic/kanjidic2.xml.gz "$dict/kanjidic2.xml.gz"
# fastText language identification, compressed model (CC BY-SA 3.0).
fetch https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz resources/lid.176.ftz

# CMU Pronouncing Dictionary for the English labels (into the default NLTK data directory).
python -c "import nltk; nltk.download('cmudict', quiet=True)"

(cd "$dict" && sha256sum -c SHA256SUMS) || echo "warning: dictionary hashes differ from the pinned versions (see above)"
echo "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83  resources/lid.176.ftz" | sha256sum -c - \
  || echo "warning: lid.176.ftz differs from the pinned model"
echo "set ECHORA_KARA_LID_MODEL=$(pwd)/resources/lid.176.ftz"
