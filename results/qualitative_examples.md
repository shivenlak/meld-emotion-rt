# Qualitative examples (MELD test, chosen by rule, see scripts/qualitative_examples.py)

Over the whole test set (ASR transcripts, seed 0), compared with the text-only head, the fused head **fixed 145** clips (in 98 of them the audio head alone also had the right answer) and **broke 139** clips that text-only got right (out of 2610). The net gain is small: audio is a real but noisy signal.

## tone_overrides_words: `test:dia199_utt13`

- **Gold:** Rachel: "I guess I just figured that somewhere down the road, we would be on again." → **sadness**
- **ASR heard:** "I guess I just figured that somewhere down the road we would be on again."
- **Text-only head:** neutral · **Audio head:** sadness · **Fused:** sadness (0.64)
- **Lumi:** Hmm, that feels sad. Maybe we'll be on again someday.
- Run it: `python run_file.py test:dia199_utt13`

## tone_overrides_words: `test:dia191_utt5`

- **Gold:** Gunther: "Do you remember when you first came here, how you spent two weeks getting trained by another waitress?" → **neutral**
- **ASR heard:** "You remember when you first came here how you spent two weeks getting trained by another waitress?"
- **Text-only head:** surprise · **Audio head:** neutral · **Fused:** neutral (0.59)
- **Lumi:** I remember when I first came here. It felt like two weeks, but I was just learning.
- Run it: `python run_file.py test:dia191_utt5`

## correct_confident: `test:dia45_utt4`

- **Gold:** Phoebe: "Oh that's so exciting!" → **joy**
- **ASR heard:** "Oh, that's so exciting! Thank you! Thanks, Ma-"
- **Text-only head:** joy · **Audio head:** joy · **Fused:** joy (0.99)
- **Lumi:** Oh, that's so exciting! I'm glad you said that. Let me know if you need anything else.
- Run it: `python run_file.py test:dia45_utt4`

## low_confidence: `test:dia203_utt8`

- **Gold:** Bonnie: "Y'know what, I should do it." → **joy**
- **ASR heard:** "You know what? I can't do it."
- **Text-only head:** sadness · **Audio head:** surprise · **Fused:** anger (0.18)
- **Lumi:** Hmm, that sounds tough. What's the thing you're trying to do? Let me know and I'll try my best!
- Run it: `python run_file.py test:dia203_utt8`

## failure: `test:dia107_utt7`

- **Gold:** Ross: "Yeah." → **sadness**
- **ASR heard:** "Mm."
- **Text-only head:** neutral · **Audio head:** joy · **Fused:** neutral (0.96)
- **Lumi:** Mmm... maybe they're thinking about something. Let me know if you want to explore that!
- Run it: `python run_file.py test:dia107_utt7`
