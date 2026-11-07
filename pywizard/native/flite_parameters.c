#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "flite.h"
#include "cst_cg.h"

typedef struct {
    const char *id;
    cst_voice *(*register_voice)(const char *);
    void (*unregister_voice)(cst_voice *);
} VoiceEntry;

#include "flite_voices.h"

static int emitted = 0;
static const VoiceEntry *selected_voice = &voices[0];
static double speed = 1.0;
static cst_utterance *active_utterance = NULL;
static int process_tone_numbers = 0;

static cst_utterance *tokenize_tones(cst_utterance *utterance)
{
    cst_item *token;
    int group = 0;
    default_tokenization(utterance);
    for (token = utt_rel_head(utterance, "Token"); token; token = item_next(token)) {
        const char *name = item_feat_string(token, "name");
        size_t length = strlen(name), i;
        char *spelling;
        ++group;
        if (length < 2 || name[length - 1] < '1' || name[length - 1] > '5') continue;
        for (i = 0; i < length - 1; ++i) {
            if (!((name[i] >= 'a' && name[i] <= 'z') ||
                  (name[i] >= 'A' && name[i] <= 'Z'))) break;
        }
        if (i != length - 1) continue;
        item_set_int(token, "tone", name[length - 1] - '0');
        item_set_int(token, "tone_group", group);
        spelling = cst_strdup(name);
        spelling[length - 1] = '\0';
        item_set_string(token, "name", spelling);
        cst_free(spelling);
    }
    return utterance;
}

static int frame_is_voiced(const cst_item *frame)
{
    const char *phone = ffeature_string(frame, "R:mcep_link.parent.R:segstate.parent.name");
    const char *vowel = ffeature_string(frame, "R:mcep_link.parent.R:segstate.parent.ph_vc");
    if (strcmp(phone, "pau") == 0) return 0;
    return strcmp(vowel, "+") == 0 || item_feat_float(frame, "voicing") > 0.5;
}

static cst_utterance *synthesize_parameters(cst_utterance *utterance)
{
    active_utterance = utterance;
    cg_synth(utterance);
    active_utterance = NULL;
    return utterance;
}

/* Replace the waveform vocoder at link time, retaining Flite's MLPG output. */
cst_wave *mlsa_resynthesis(const cst_track *params, const cst_track *strength,
                          cst_cg_db *db, cst_audio_streaming_info *stream,
                          int speed)
{
    int i, j;
    cst_wave *empty;
    const cst_item *frame = active_utterance ? utt_rel_head(active_utterance, "mcep") : NULL;
    (void)strength;
    (void)stream;
    (void)speed;
    printf("{\"version\":2,\"sample_rate\":%d,\"alpha\":%.9g,"
           "\"frame_shift\":%.9g,\"frames\":[",
           db->sample_rate, db->mlsa_alpha, db->frame_advance);
    for (i = 0; i < params->num_frames; ++i) {
        if (i) putchar(',');
        putchar('[');
        for (j = 0; j < params->num_channels; ++j) {
            float value = params->frames[i][j];
            /* SPAM F0 voices overwrite the earlier zero-F0 voicing mask.
               Restore Flite's original decision before encoding chip frames. */
            if (j == 0 && frame && !frame_is_voiced(frame)) value = 0.0;
            if (j) putchar(',');
            printf("%.9g", value);
        }
        putchar(']');
        if (frame) frame = item_next(frame);
    }
    putchar(']');
    if (process_tone_numbers) {
        const char *token_path = "R:mcep_link.parent.R:segstate.parent.R:SylStructure.parent.parent.R:Token.parent";
        frame = utt_rel_head(active_utterance, "mcep");
        printf(",\"tone_marks\":[");
        for (i = 0; i < params->num_frames; ++i) {
            const cst_item *token = frame ? path_to_item(frame, token_path) : NULL;
            if (i) putchar(',');
            printf("[%d,%d]", token ? get_param_int(item_feats(token), "tone_group", 0) : 0,
                   token ? get_param_int(item_feats(token), "tone", 0) : 0);
            if (frame) frame = item_next(frame);
        }
        putchar(']');
    }
    puts("}");
    emitted = 1;
    empty = new_wave();
    empty->sample_rate = db->sample_rate;
    return empty;
}

int main(int argc, char **argv)
{
    size_t length = 0, capacity = 4096;
    int ch, argument;
    size_t index;
    char *text;
    cst_voice *voice;
    cst_utterance *utterance;
    for (argument = 1; argument < argc; ++argument) {
        if (strcmp(argv[argument], "--voice") == 0 && argument + 1 < argc) {
            const char *id = argv[++argument];
            selected_voice = NULL;
            for (index = 0; index < sizeof(voices) / sizeof(voices[0]); ++index) {
                if (strcmp(id, voices[index].id) == 0) selected_voice = &voices[index];
            }
            if (!selected_voice) {
                fprintf(stderr, "Unknown Flite voice: %s. Rebuild the helper if voices changed.\n", id);
                return 2;
            }
        } else if (strcmp(argv[argument], "--speed") == 0 && argument + 1 < argc) {
            char *end;
            speed = strtod(argv[++argument], &end);
            if (*end || !isfinite(speed) || speed < 0.5 || speed > 2.0) {
                fputs("Speed must be between 0.5 and 2.0.\n", stderr);
                return 2;
            }
        } else if (strcmp(argv[argument], "--process-tone-numbers") == 0) {
            process_tone_numbers = 1;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[argument]);
            return 2;
        }
    }
    text = malloc(capacity);
    if (!text) return 1;
    while ((ch = getchar()) != EOF) {
        if (length + 1 >= capacity) {
            char *larger;
            capacity *= 2;
            larger = realloc(text, capacity);
            if (!larger) { free(text); return 1; }
            text = larger;
        }
        text[length++] = (char)ch;
    }
    text[length] = '\0';
    if (!length) { free(text); return 2; }
    flite_init();
    voice = selected_voice->register_voice(NULL);
    if (process_tone_numbers)
        flite_feat_set(voice->features, "tokenizer_func", uttfunc_val(&tokenize_tones));
    flite_feat_set_float(voice->features, "duration_stretch", 1.0 / speed);
    flite_feat_set(voice->features, "wave_synth_func", uttfunc_val(&synthesize_parameters));
    utterance = flite_synth_text(text, voice);
    if (utterance) delete_utterance(utterance);
    selected_voice->unregister_voice(voice);
    free(text);
    return emitted ? 0 : 3;
}
