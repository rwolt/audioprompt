"""UI translation tables for the AudioPrompt Streamlit app.

Gettext-style: the English UI literal is the lookup key, so any string missing
from a table silently falls back to English. Keys must match the literals in
app.py exactly (including unicode dashes/quotes); tests/check_i18n_coverage.py
verifies coverage by extracting the literals from app.py's AST.

Translation notes (ja): drafted by Claude; natural-sounding UI Japanese using
standard katakana audio-production loanwords. Native-speaker corrections are
very welcome — every string lives in this one file.
"""
from __future__ import annotations

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "ja": "日本語",
}

JA: dict[str, str] = {
    # ---- top controls (rendered by i18n.py) ----
    "Show help as visible text": "ヘルプをテキストで表示",
    "Shows each control's explanation as plain text under the control instead of a hover tooltip. Recommended for screen readers; also lets the language setting translate every explanation.":
        "各コントロールの説明を、ホバー式ツールチップではなくコントロールの下にテキストとして表示します。スクリーンリーダー使用時に推奨。言語設定による説明の翻訳もこのモードで確実になります。",
    "Screen reader tip: check the 'Show help as visible text' checkbox below to read every control's explanation as regular text instead of hover-only tooltips.":
        "スクリーンリーダーをお使いの方へ: この下の「ヘルプをテキストで表示」チェックボックスをオンにすると、各コントロールの説明がホバー式ツールチップではなく通常のテキストとして読めるようになります。",

    # ---- section headers ----
    "Quick Start": "クイックスタート",
    "Input Audio": "入力オーディオ",
    "Melody": "メロディ",
    "Drum MIDI Imprint": "ドラムMIDIインプリント",
    "Bass MIDI Imprint": "ベースMIDIインプリント",
    "Focus": "フォーカス",
    "Generate": "生成",
    "Outputs": "出力",

    # ---- input & sample rate ----
    "Input audio (optional)": "入力オーディオ（任意）",
    "Drag & drop a file. If provided, the prompt will be prepended to create a combined output. MP3 support depends on your libsndfile build.":
        "ファイルをドラッグ＆ドロップ。指定するとプロンプトを先頭に連結した結合出力を作成します。MP3対応はlibsndfileのビルドに依存します。",
    "Sample rate (Hz)": "サンプルレート (Hz)",
    "Processing rate; inputs are resampled. Higher SR costs more CPU.":
        "処理サンプルレート。入力はリサンプリングされます。高いほどCPU負荷が上がります。",

    # ---- melody ----
    "Enable melody": "メロディを有効化",
    "Imprint a randomized melody (scale‑constrained) onto pink noise.":
        "スケールに沿ったランダムメロディをピンクノイズにインプリントします。",
    "Rhythmic gate": "リズミックゲート",
    "Apply a note‑shaped amplitude envelope for phrasing.":
        "フレージングのため、ノート形状の音量エンベロープを適用します。",
    "Root": "ルート",
    "Root note for the scale (C4=60).": "スケールのルート音 (C4=60)。",
    "Scale": "スケール",
    "Choose from major/modes, pentatonics, blues, etc.":
        "メジャー／モード、ペンタトニック、ブルースなどから選択。",
    "BPM": "BPM",
    "Tempo driving randomized note durations. Go as low as 20 for halftime vocals over 40 BPM drum tracks.":
        "ランダムなノート長を決めるテンポ。40 BPMのドラムにハーフタイムのボーカルを乗せる場合などは20まで下げられます。",
    "Vibrato Hz": "ビブラート (Hz)",
    "Rate of pitch modulation.": "ピッチ変調の速さ。",
    "Vibrato depth": "ビブラートの深さ",
    "Depth of pitch modulation (fraction).": "ピッチ変調の深さ（割合）。",
    "Melody character": "メロディの音色",
    "Harmonic color of the melody imprint. Neutral: flat harmonic stack. Warm: rolls off high harmonics for a mellow tone. Voice: emphasizes low harmonics (1–5) for a vocal buzz. Reed: odd harmonics only (clarinet-like). Bell: peaks around the 3rd harmonic for a metallic shimmer. Pluck: fast harmonic decay. Bright: boosts higher harmonics for more edge. Wide: neutral harmonics + 7-cent detune spread for subtle thickness.":
        "メロディインプリントの倍音カラー。ニュートラル: フラットな倍音構成。ウォーム: 高次倍音を抑えたまろやかな音。ボイス: 低次倍音（1〜5）を強調した声のようなバズ。リード: 奇数倍音のみ（クラリネット風）。ベル: 第3倍音付近にピークを持つ金属的な響き。プラック: 倍音が速く減衰。ブライト: 高次倍音を強調したエッジのある音。ワイド: ニュートラルな倍音＋7セントのデチューンで厚みを追加。",
    "Note shape": "ノートシェイプ",
    "Shapes the base envelope for the melody notes. Tight and pluck have exponential decays; smooth has a softer attack.":
        "メロディノートの基本エンベロープ。tightとpluckは指数減衰、smoothはアタックが柔らかめです。",
    "Note decay": "ノートディケイ",
    "Scales the length of the note envelope decay. Lower = shorter, staccato notes.":
        "ノートエンベロープの減衰の長さ。低いほど短くスタッカートに。",
    "Melody – Advanced": "メロディ – 詳細設定",
    "Low MIDI": "下限MIDI",
    "Register floor. Defaults to the root note in octave 3; adjust to taste.":
        "音域の下限。デフォルトはオクターブ3のルート音。好みに応じて調整してください。",
    "Step bias": "ステップ傾向",
    "Probability of moving to a neighboring scale degree.": "隣のスケール音へ進む確率。",
    "Glide prob": "グライド確率",
    "Probability of sliding into the next note.": "次のノートへスライドして入る確率。",
    "High MIDI": "上限MIDI",
    "Register ceiling. Defaults to 1.5 octaves above Low MIDI.":
        "音域の上限。デフォルトは下限MIDIの1.5オクターブ上。",
    "Max leap (scale steps)": "最大跳躍（スケール度数）",
    "Largest jump when not stepping.": "ステップ移動しないときの最大ジャンプ幅。",
    "Glide frac": "グライド割合",
    "Portion of the note duration spent gliding.": "ノート長のうちグライドに使う割合。",
    "Rest prob": "休符確率",
    "Chance of rests vs notes.": "ノートに対する休符の確率。",
    "Root drone level": "ルートドローン量",
    "Subtle sustained drone on the root note beneath the melody.":
        "メロディの下で控えめに持続するルート音のドローン。",
    "Imprint gain": "インプリントゲイン",
    "Strength of harmonic emphasis on the pink noise.": "ピンクノイズへの倍音強調の強さ。",
    "Harmonics": "倍音数",
    "Number of harmonic peaks in the pitch mask.": "ピッチマスク内の倍音ピーク数。",
    "BW frac": "帯域幅比 (BW frac)",
    "Primary texture control. Narrow (0.005–0.01): tight pitch instruction, can sound synthetic. Widen (0.02–0.05) to make the melody feel less robotic without losing the harmonic steer. Go higher (0.05–0.1) for a very diffuse, atmospheric texture.":
        "主要なテクスチャ調整。狭い (0.005〜0.01): タイトなピッチ指示だがやや人工的。広め (0.02〜0.05): 倍音の誘導を保ちつつロボット感を軽減。さらに広く (0.05〜0.1): 非常に拡散したアンビエントな質感に。",
    "Noise floor (dB)": "ノイズフロア (dB)",
    "Attenuation of non-harmonic pink noise. 0 dB = full noise floor (current behavior). Lower values reduce broadband static while keeping harmonic emphasis. Try –18 to –30 dB if the prompt is introducing extra noise into your AI music output.":
        "非倍音成分のピンクノイズの減衰量。0 dB = ノイズフロアそのまま（従来動作）。下げるほど倍音強調を保ったまま広帯域ノイズを削減。AI出力にノイズが乗る場合は −18〜−30 dB を試してください。",

    # ---- vowel character ----
    "Vowel character": "母音キャラクター",
    "Imprint vowels": "母音をインプリント",
    "Shape the melody imprint toward sung vowel sounds in the selected language. Off (default) = a vowel-neutral buzz, which gives the AI more freedom and often hallucinates instruments better. On = biases the output toward voice-like vowel color. This adds vowel resonances only — no consonants, no words; it cannot make the AI sing specific lyrics or guarantee a language.":
        "選択した言語の歌唱母音に向けてメロディインプリントを整形します。オフ（デフォルト）= 母音を持たないバズ音で、AIの自由度が高く楽器の生成に向くことが多いです。オン = 声らしい母音の色付けへ誘導します。付加されるのは母音の共鳴のみで、子音や単語は含まれません。特定の歌詞を歌わせたり、言語を保証したりはできません。",
    "Language": "言語",
    "Vowel system used for the imprint. English: full vowels on stressed notes, weak notes reduce to a neutral 'schwa' — the rhythm signature of stress-timed English. Japanese: the five vowels a-i-u-e-o on every note with no reduction (Japanese is mora-timed and keeps every vowel full; its 'u' is the brighter unrounded Japanese u). Spanish: the five Spanish vowels on every note, no reduction (syllable-timed).":
        "インプリントに使う母音体系。英語: 強勢のあるノートに完全母音、弱いノートは曖昧母音（シュワー）に弱化 — 強勢拍リズムである英語の特徴。日本語: すべてのノートに「あ・い・う・え・お」を使い弱化なし（モーラ拍の日本語は母音が常に明瞭。「う」は非円唇の明るい日本語のウ）。スペイン語: 5つの母音をすべてのノートに使用、弱化なし（音節拍）。",
    "Vowel strength": "母音の強さ",
    "How hard the vowels are imprinted. Low = a hint the model can override; high = strong vowel character that can sound robotic if overdone. Start ~0.6 and tune by ear.":
        "母音インプリントの強さ。低 = モデルが上書きできるヒント程度。高 = 強い母音キャラクター（やり過ぎるとロボット的に）。0.6前後から耳で調整してください。",
    "Stress pattern": "強勢パターン",
    "English only. Accented notes get full vowels; the rest reduce to a neutral 'schwa'. 2 = strong-weak (most English-like), 3 = strong-weak-weak (more lilting). Vowel reduction is the acoustic signature of stress-timed English — Japanese and Spanish keep every vowel full, so this control is hidden for them.":
        "英語のみ。アクセントのあるノートは完全母音、それ以外は曖昧母音（シュワー）に弱化します。2 = 強弱（最も英語的）、3 = 強弱弱（より軽快）。母音弱化は強勢拍である英語の音響的特徴です — 日本語とスペイン語は全母音が明瞭なため、このコントロールは表示されません。",
    "A gentle acoustic bias, not a lyric engine — it colors the noise toward the language's vowel sounds, but the AI tool's own text prompt still has the biggest influence on language and words.":
        "これは緩やかな音響的バイアスであり、歌詞エンジンではありません。ノイズを選択言語の母音の響きに寄せますが、言語や歌詞への影響はAIツール側のテキストプロンプトの方がはるかに大きいです。",

    # ---- drum MIDI ----
    "Enable drum MIDI imprint": "ドラムMIDIインプリントを有効化",
    "Upload a MIDI drum track to generate a rhythm layer from band-limited pink noise.":
        "MIDIドラムトラックをアップロードして、帯域制限ピンクノイズによるリズムレイヤーを生成します。",
    "Drum MIDI (.mid / .midi)": "ドラムMIDI (.mid / .midi)",
    "Upload a General MIDI drum track exported from your DAW or drum sequencer. Use MIDI note numbers as the source of truth: kick 36, snare 38, closed hat 42. Your DAW may label those as C1/D1/F#1 or C2/D2/F#2 depending on octave numbering.":
        "DAWやドラムシーケンサーから書き出したGeneral MIDIドラムトラックをアップロード。MIDIノート番号が基準です: キック36、スネア38、クローズドハット42。DAWによってはC1/D1/F#1やC2/D2/F#2と表示されます。",
    "Kick amount": "キック量",
    "Gain for General MIDI kick notes 35/36. Your DAW may display MIDI 36 as C1 or C2 depending on octave numbering.":
        "General MIDIキックノート35/36のゲイン。DAWによりMIDI 36はC1またはC2と表示されます。",
    "Snare amount": "スネア量",
    "Gain for General MIDI snare, clap, side stick, and tom notes routed to the snare noise lane.":
        "スネアノイズレーンにルーティングされるGeneral MIDIのスネア、クラップ、サイドスティック、タムのゲイン。",
    "Hat amount": "ハット量",
    "Gain for General MIDI hi-hat and cymbal notes routed to the hat noise lane.":
        "ハットノイズレーンにルーティングされるGeneral MIDIのハイハットとシンバルのゲイン。",
    "Perc amount": "パーカッション量",
    "Gain for miscellaneous General MIDI percussion and any unmapped notes routed to the perc noise lane.":
        "パーカッションレーンにルーティングされるその他のGeneral MIDIパーカッションと未マッピングノートのゲイン。",
    "Drum character": "ドラムの音色",
    "Tonal preset for the drum layer. Clean: balanced default. Tight: shorter envelopes and snappier transients. Deep: bass-heavy low end with extra kick weight. Bright: more hat and upper-mid presence. Breakbeat: pumped body, snappy snare, and light drive — groove timing stays intact.":
        "ドラムレイヤーの音色プリセット。クリーン: バランスの取れたデフォルト。タイト: 短いエンベロープとキレのあるトランジェント。ディープ: キックを強めた重低音。ブライト: ハットと中高域を強調。ブレイクビーツ: 張りのあるボディ、スナッピーなスネア、軽いドライブ — グルーヴのタイミングは維持されます。",
    "Snare tune": "スネアチューン",
    "Shifts the snare lane's noise band in semitones before the drum layer is synthesized.":
        "ドラムレイヤーの合成前に、スネアレーンのノイズ帯域を半音単位でシフトします。",
    "Drum decay": "ドラムディケイ",
    "Shortens or lengthens all drum lane envelopes. Lower values are tighter; higher values ring longer.":
        "全ドラムレーンのエンベロープ長を調整。低い = タイト、高い = 余韻長め。",
    "Match melody BPM": "メロディBPMに合わせる",
    "Use the melody BPM for the drum layer. Turn off to set an independent drum BPM.":
        "ドラムレイヤーにメロディBPMを使用。オフにすると独立したドラムBPMを設定できます。",
    "Use detected BPM": "検出BPMを使用",
    "Reset Independent drum BPM to the tempo detected in the uploaded MIDI file.":
        "ドラム独立BPMを、アップロードしたMIDIから検出したテンポにリセットします。",
    "Independent drum BPM": "ドラム独立BPM",
    "Target BPM used only when Match melody BPM is off. New uploads start at their detected MIDI BPM so you can quickly return to the original groove tempo.":
        "「メロディBPMに合わせる」がオフのときのみ使うターゲットBPM。新しいアップロードは検出BPMから始まるため、元のグルーヴのテンポへすぐ戻れます。",
    "Loop drums to prompt length": "プロンプト長までドラムをループ",
    "Repeats the drum MIDI when the prompt is longer than the MIDI region. Turn off to let the drums stop after the uploaded MIDI ends.":
        "プロンプトがMIDIリージョンより長い場合にドラムMIDIを繰り返します。オフにするとMIDI終了後にドラムが止まります。",
    "Trim silence before first note": "最初のノート前の無音をカット",
    "Removes empty lead-in before the first drum hit (e.g. Logic session-player exports add a phantom bar). Turn off if your drums genuinely start with deliberate leading silence.":
        "最初のドラムヒット前の空白をカットします（Logicのセッションプレイヤー書き出しなどで入る余分な小節への対策）。意図的な先頭無音がある場合はオフにしてください。",

    # ---- bass MIDI ----
    "Enable bass MIDI imprint": "ベースMIDIインプリントを有効化",
    "Upload a MIDI bassline to generate a nuanced, pitch-sliding bass layer.":
        "MIDIベースラインをアップロードして、ピッチスライドを含む繊細なベースレイヤーを生成します。",
    "Bass MIDI (.mid / .midi)": "ベースMIDI (.mid / .midi)",
    "Upload a bass track. Pitch bend and note velocity will be preserved and imprinted.":
        "ベーストラックをアップロード。ピッチベンドとベロシティが保持されインプリントされます。",
    "Bass character": "ベースの音色",
    "Tonal preset for the bass imprint. Upright: warm, pluck-like decay in the low-mid range (40–800 Hz). Fingerstyle: balanced midrange tone (40–2000 Hz). Picked: bright with lots of harmonics and strong attack transient (40–5000 Hz). Sub: deep, smooth, low-frequency-only (30–200 Hz). Synth: odd-harmonic reed-like character with smooth envelope (30–3000 Hz).":
        "ベースインプリントの音色プリセット。アップライト: 低中域 (40〜800 Hz) の温かいプラック風減衰。フィンガー: バランスの取れた中域 (40〜2000 Hz)。ピック: 倍音豊富で強いアタック (40〜5000 Hz)。サブ: 深く滑らかな低域のみ (30〜200 Hz)。シンセ: 奇数倍音のリード的キャラクターと滑らかなエンベロープ (30〜3000 Hz)。",
    "Note shape base": "ノートシェイプ基本形",
    "Base envelope shape before Decay offset and velocity are applied.":
        "ディケイオフセットとベロシティが適用される前の基本エンベロープ形状。",
    "Decay offset": "ディケイオフセット",
    "Scales the length of the bass note envelope decay.":
        "ベースノートのエンベロープ減衰の長さを調整します。",
    "Pitch bend range": "ピッチベンドレンジ",
    "Matches the pitch wheel range of the virtual instrument that generated the MIDI (Logic slides often use 12, 24, or 48 semitones).":
        "MIDIを生成した音源のピッチホイールレンジに合わせてください（Logicのスライドは12、24、48半音が多いです）。",
    "Match melody BPM (Bass)": "メロディBPMに合わせる（ベース）",
    "Use the melody BPM for the bass layer.": "ベースレイヤーにメロディBPMを使用します。",
    "Reset Independent bass BPM to the tempo detected in the uploaded MIDI file.":
        "ベース独立BPMを、アップロードしたMIDIから検出したテンポにリセットします。",
    "Independent bass BPM": "ベース独立BPM",
    "Loop bass to prompt length": "プロンプト長までベースをループ",
    "Repeats the bass MIDI when the prompt is longer than the MIDI region.":
        "プロンプトがMIDIリージョンより長い場合にベースMIDIを繰り返します。",
    "Removes empty lead-in before the first note (e.g. Logic session-player exports add a phantom bar). Preserves an intentional pickup or rest that's written on the note itself. Turn off if your bass should start with deliberate leading silence.":
        "最初のノート前の空白をカットします（Logicのセッションプレイヤー書き出しなどで入る余分な小節への対策）。ノート自体に書かれたピックアップや休符は保持されます。意図的な先頭無音がある場合はオフにしてください。",
    "Bass – Advanced": "ベース – 詳細設定",
    "Bass imprint gain": "ベースインプリントゲイン",
    "Strength of harmonic emphasis on the bass pitch mask.":
        "ベースピッチマスクへの倍音強調の強さ。",
    "Bass BW frac": "ベース帯域幅比 (BW frac)",
    "Relative bandwidth around each harmonic. Wider = looser, less synthetic-sounding pitch lock.":
        "各倍音まわりの相対帯域幅。広いほど緩く、人工的でないピッチロックになります。",
    "Bass noise floor (dB)": "ベースノイズフロア (dB)",
    "Attenuation of non-harmonic noise within the bass focus band. 0 dB = no attenuation (current behavior). Lower values reduce residual static inside the bass frequency window. Less impactful than melody noise floor since the bass focus band is already narrow.":
        "ベースフォーカス帯域内の非倍音ノイズの減衰量。0 dB = 減衰なし（従来動作）。下げるほど帯域内の残留ノイズを削減。帯域が狭いため、メロディのノイズフロアほどの効果はありません。",

    # ---- focus ----
    "Enable focus band": "フォーカス帯域を有効化",
    "Emphasize energy in a vocal/guitar/bass band or a custom Hz range.":
        "ボーカル／ギター／ベース帯域、またはカスタムHz範囲のエネルギーを強調します。",
    "Bass Roll-Off": "ベースロールオフ",
    "Applies a 25 Hz high-pass filter to the generated prompt noise before melody/focus imprinting. This removes sub-bass rumble from the prompt layer only; it does not EQ your uploaded audio.":
        "メロディ／フォーカスのインプリント前に、生成ノイズへ25 Hzのハイパスフィルタを適用します。プロンプトレイヤーのサブベースのみを除去し、アップロード音声にはEQをかけません。",
    "Preset": "プリセット",
    "Choose a preset band or select ‘custom’ to set your own Hz range.":
        "プリセット帯域を選ぶか、「custom」で独自のHz範囲を設定します。",
    "Focus Band – Advanced": "フォーカス帯域 – 詳細設定",
    "Focus Hz band": "フォーカス帯域 (Hz)",
    "Twin‑handle slider: low/high cutoff in Hz.":
        "ツインハンドルスライダー: 低域／高域カットオフ (Hz)。",
    "Band floor (dB)": "帯域外フロア (dB)",
    "Attenuation outside the focus band.": "フォーカス帯域外の減衰量。",
    "Band edge sharpness": "帯域エッジの鋭さ",
    "Sigmoid steepness of the band rolloff. 6 = gentle slope; 12 = moderate; 24 ≈ near-brick-wall. Dimensionless — not dB/Hz.":
        "帯域ロールオフのシグモイド急峻度。6 = 緩やか、12 = 中程度、24 ≈ ほぼブリックウォール。無次元 — dB/Hzではありません。",

    # ---- generate / output ----
    "Generate Prompt": "プロンプトを生成",
    "Seed": "シード",
    "Controls randomness for pink noise and the melody (notes, glides, etc.). Set to -1 to use a new random seed each generation.":
        "ピンクノイズとメロディ（ノート、グライドなど）のランダム性を制御します。-1にすると毎回新しいランダムシードを使用します。",
    "Prompt length": "プロンプト長",
    "Match length to uploaded MIDI regions (adjusted to target BPM) or use manual seconds.":
        "アップロードしたMIDIリージョンの長さに合わせる（ターゲットBPMに調整）か、秒数を手動で指定します。",
    "Manual seconds": "秒数を指定",
    "Prompt seconds": "プロンプト秒数",
    "Match drum MIDI": "ドラムMIDIに合わせる",
    "Match bass MIDI": "ベースMIDIに合わせる",
    "Length of the generated prompt when Prompt length is set to Manual seconds.":
        "プロンプト長が「秒数を指定」のときの生成プロンプトの長さ。",
    "Length of the generated prompt.": "生成するプロンプトの長さ。",
    " Note: the public demo caps prompt length at {cap} s.":
        " ※公開デモではプロンプト長の上限は{cap}秒です。",
    "This prompt would be {length}s — the public demo caps prompts at {cap}s (plenty for a 16-bar loop even at 70 BPM). Trim your MIDI to {cap}s or shorter, or run the app yourself for unlimited length: https://github.com/rwolt/audioprompt":
        "このプロンプトは{length}秒になりますが、公開デモの上限は{cap}秒です（70 BPMの16小節ループでも十分収まります）。MIDIを{cap}秒以内にトリミングするか、長さ無制限で使うにはアプリをご自身で実行してください: https://github.com/rwolt/audioprompt",
    "Auto gain (match input audio)": "オートゲイン（入力に合わせる）",
    "Detect input audio loudness and set the prompt level automatically (RMS in dBFS).":
        "入力音声のラウドネスを検出し、プロンプトレベルを自動設定します（RMS dBFS）。",
    "Prompt relative to input (dB)": "入力に対するプロンプト (dB)",
    "Target prompt loudness relative to input RMS (e.g., −3 dB makes the prompt slightly quieter).":
        "入力RMSに対するプロンプトの目標ラウドネス（例: −3 dBでプロンプトをやや小さく）。",
    "Prompt gain (dB)": "プロンプトゲイン (dB)",
    "Level for the prepended prompt.": "先頭に連結するプロンプトのレベル。",
    "Fade-in (ms)": "フェードイン (ms)",
    "Smooth ramp at the start.": "開始部分のスムーズなランプ。",
    "Fade-out (ms)": "フェードアウト (ms)",
    "Smooth ramp at the end.": "終了部分のスムーズなランプ。",
    "Melody level": "メロディレベル",
    "Gain for the melody / focus / pink-noise layer, applied after loudness-matching all layers to the same RMS level.":
        "全レイヤーを同一RMSにラウドネスマッチした後に適用される、メロディ／フォーカス／ピンクノイズレイヤーのゲイン。",
    "Drum level": "ドラムレベル",
    "Gain for the drum MIDI layer, applied after loudness-matching all layers to the same RMS level.":
        "全レイヤーを同一RMSにラウドネスマッチした後に適用される、ドラムMIDIレイヤーのゲイン。",
    "Bass level": "ベースレベル",
    "Gain for the bass MIDI layer, applied after loudness-matching all layers to the same RMS level.":
        "全レイヤーを同一RMSにラウドネスマッチした後に適用される、ベースMIDIレイヤーのゲイン。",
    "Generating prompt...": "プロンプトを生成中...",
    "Prompt generated — {length} sec": "プロンプトを生成しました — {length}秒",
    "Download Prompt": "プロンプトをダウンロード",
    "Download Combined": "結合版をダウンロード",
    "**Prompt Level**": "**プロンプトレベル**",
    "**Layer Blend**": "**レイヤーブレンド**",
    "**Prompt**": "**プロンプト**",
    "**Combined**": "**結合版**",
    "---": "---",

    # ---- dynamic captions / messages ----
    "Uploaded: {name}": "アップロード済み: {name}",
    "Detected drum MIDI: {bpm} BPM, {length} sec": "検出したドラムMIDI: {bpm} BPM、{length}秒",
    "Detected bass MIDI: {bpm} BPM, {length} sec": "検出したベースMIDI: {bpm} BPM、{length}秒",
    "Matched drum length: {length} sec at {bpm} BPM": "{bpm} BPMでのドラム長: {length}秒",
    "Matched bass length: {length} sec at {bpm} BPM": "{bpm} BPMでのベース長: {length}秒",
    "Prompt length matched to drum MIDI: {length} sec": "プロンプト長をドラムMIDIに一致: {length}秒",
    "Prompt length matched to bass MIDI: {length} sec": "プロンプト長をベースMIDIに一致: {length}秒",
    "Seed used: {seed}": "使用シード: {seed}",
    "Drum layer: {bpm} BPM (detected {detected} BPM)": "ドラムレイヤー: {bpm} BPM（検出 {detected} BPM）",
    "Drum layer: {bpm} BPM": "ドラムレイヤー: {bpm} BPM",
    "Bass layer: {bpm} BPM (detected {detected} BPM)": "ベースレイヤー: {bpm} BPM（検出 {detected} BPM）",
    "Bass layer: {bpm} BPM": "ベースレイヤー: {bpm} BPM",
    "Failed to parse drum MIDI. Make sure you uploaded a valid .mid/.midi file.\nError: {err}":
        "ドラムMIDIの解析に失敗しました。有効な .mid / .midi ファイルか確認してください。\nエラー: {err}",
    "Failed to parse bass MIDI. Make sure you uploaded a valid .mid/.midi file.\nError: {err}":
        "ベースMIDIの解析に失敗しました。有効な .mid / .midi ファイルか確認してください。\nエラー: {err}",
    "Failed to read input audio. Prefer WAV/FLAC/OGG. MP3 support depends on your libsndfile build.\nError: {err}":
        "入力オーディオの読み込みに失敗しました。WAV/FLAC/OGGを推奨します。MP3対応はlibsndfileのビルドに依存します。\nエラー: {err}",
    "Set your parameters and press Generate Prompt.":
        "パラメータを設定して「プロンプトを生成」を押してください。",
    "No input file uploaded; only the prompt is generated.":
        "入力ファイルが無いため、プロンプトのみ生成されます。",

    # ---- option value display names ----
    "neutral": "ニュートラル", "warm": "ウォーム", "voice": "ボイス", "reed": "リード",
    "bell": "ベル", "pluck": "プラック", "bright": "ブライト", "wide": "ワイド",
    "natural": "ナチュラル", "tight": "タイト", "smooth": "スムーズ",
    "clean": "クリーン", "deep": "ディープ", "breakbeat": "ブレイクビーツ",
    "Upright": "アップライト", "Fingerstyle": "フィンガー", "Picked": "ピック",
    "Synth": "シンセ", "Sub": "サブ",
    "vocal": "ボーカル", "guitar": "ギター", "bass": "ベース", "custom": "カスタム",
    "English": "英語", "Japanese": "日本語", "Spanish": "スペイン語",
}

TABLES: dict[str, dict[str, str]] = {"ja": JA}

# Multi-line content blocks, keyed by name then language. Kept out of the
# literal-keyed table so indentation/formatting changes in app.py can't break
# the lookup.
BLOCKS: dict[str, dict[str, str]] = {
    "quick_start": {
        "en": """
AudioPrompt creates a short, steerable pink‑noise clip that can guide AI music models. It imprints a scale‑based melody, adds optional drum and bass layers from your MIDI files, emphasizes a frequency band, and can prepend the prompt to your input audio.

1. (Optional) Drag‑drop input audio — the prompt will be **prepended** to it to create a combined WAV. Leave empty for a prompt-only WAV. An instrument sample (a short synth, guitar, or piano phrase) works great as a starting point.
2. Choose Melody settings (root/scale/BPM). Optionally enable the Drum and Bass MIDI Imprint sections and upload .mid files to add rhythm and bassline layers.
3. Use Focus (or Custom band) and keep Bass Roll-Off on for cleaner prompt starts.
4. On the right, set Prompt length — match an uploaded MIDI or use manual seconds — then click Generate Prompt. Preview the Prompt (and Combined, if input audio was provided) and download the tagged WAVs.

Tips: 3–6 s prompts give a clear steer without masking; “Vocal” focus often helps melody “speak”.
""",
        "ja": """
AudioPromptは、AI音楽モデルを誘導できる短いピンクノイズクリップ（オーディオプロンプト）を生成します。スケールに沿ったメロディをインプリントし、MIDIファイルからドラム・ベースレイヤーを追加し、特定の周波数帯域を強調し、入力オーディオの先頭にプロンプトを連結できます。

1. （任意）入力オーディオをドラッグ＆ドロップ — プロンプトが**先頭に連結**され、結合WAVが作成されます。空のままならプロンプトのみのWAVになります。楽器のサンプル（短いシンセ、ギター、ピアノのフレーズなど）が出発点として最適です。
2. メロディ設定（ルート／スケール／BPM）を選びます。必要に応じてドラム・ベースMIDIインプリントを有効にし、.midファイルをアップロードしてリズムやベースラインのレイヤーを追加します。
3. フォーカス（またはカスタム帯域）を使い、ベースロールオフをオンのままにするとプロンプトの立ち上がりがクリーンになります。
4. 右側でプロンプト長を設定し（アップロードしたMIDIに合わせるか秒数を指定）、「プロンプトを生成」をクリック。プロンプト（入力オーディオがあれば結合版も）を試聴し、タグ付きWAVをダウンロードします。

ヒント: 3〜6秒のプロンプトはマスキングを起こさずに明確に誘導できます。「ボーカル」フォーカスはメロディを「歌わせる」のに効果的です。
""",
    },
    "footer_terms": {
        "en": "Terms: Upload only content you own or have rights to. By using this app you confirm permission to process any uploaded audio.",
        "ja": "利用規約: ご自身が権利を有するコンテンツのみアップロードしてください。本アプリの利用により、アップロードした音声を処理する許可があることを確認したものとみなされます。",
    },
    "footer_links": {
        "en": (
            'This project is open source: '
            '<a href="https://github.com/rwolt/audioprompt" target="_blank" rel="noopener">GitHub repository</a> · '
            '<a href="https://github.com/rwolt/audioprompt/blob/main/audioprompt_app/README.md" target="_blank" rel="noopener">app manual</a> · '
            'Part of <a href="https://taktlabs.io" target="_blank" rel="noopener">taktlabs.io</a> →'
        ),
        "ja": (
            'このプロジェクトはオープンソースです: '
            '<a href="https://github.com/rwolt/audioprompt" target="_blank" rel="noopener">GitHubリポジトリ</a>'
            '（<a href="https://github.com/rwolt/audioprompt/blob/main/README.ja.md" target="_blank" rel="noopener">日本語README</a>）・'
            '<a href="https://github.com/rwolt/audioprompt/blob/main/audioprompt_app/README.md" target="_blank" rel="noopener">アプリマニュアル（英語）</a>・'
            '<a href="https://taktlabs.io" target="_blank" rel="noopener">taktlabs.io</a> のプロジェクト →'
        ),
    },
}
