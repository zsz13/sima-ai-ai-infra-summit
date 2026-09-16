# Object vocabulary: what the detector can and cannot see

## The class list is fixed by the weights, not by the text file

`assets/coco.txt` is a **label lookup**, not a configuration. The detector emits a
class *index*; the file only turns that index into a word.

Verified on the local YOLO26n ONNX export:

- the model's own `id2label` has exactly **80 entries**
- `assets/coco.txt` has exactly **80 lines**
- the two are **identical, index for index** (checked programmatically, not assumed)

So **adding names to `coco.txt` cannot teach the model a new object.** The output
layer has 80 channels and no more. Appending an 81st line would either be ignored
(no index ever selects it) or, if anything shifted, silently corrupt the mapping
so that every class after the insertion point is reported under the wrong name -
a `person` labelled `bicycle`, and detector grounding quietly wrong everywhere.

The 80-class mapping stays exactly as it is.

The Modalix path has the same property: `yolo_26n_mpk` is compiled with a fixed
80-class head, and the board's `coco.txt` is the same lookup.

## Confusion inside the class set is normal, not a bug

Classes that look alike share probability. For a clearly visible remote control
(48x30 px in the 640 input, correct answer, confident):

| class | score |
|---|---|
| **remote** | **0.5637** |
| cell phone | 0.0096 |
| book | 0.0071 |
| keyboard | 0.0032 |
| laptop | 0.0015 |

The runners-up are precisely the flat-rectangle family. Under worse conditions -
small, blurred, off-angle, back-lit - that ordering can invert, which is how a
book comes back as a cell phone. It is the model's learned similarity, not a
preprocessing or label-mapping fault, and Camera Check's debug view exists so it
can be seen rather than guessed at.

## Options for a broader vocabulary - assessment only, nothing installed

### 1. A larger fixed-class detector
Swap YOLO26n for a bigger variant, or a model trained on a larger taxonomy
(LVIS, ~1200 classes; Objects365, 365 classes).

- **Gains**: real detector grounding for many more objects, same architecture,
  same integration, no API change.
- **Costs**: still a fixed list, so "pen" is only covered if that list happens to
  contain it. A larger backbone costs latency in proportion.
- **Modalix**: needs a full `llima-compile` pass to an MLA `.elf`, which is the
  expensive part - the toolchain supports YOLO-family graphs, but a new
  architecture is a compile project, not a swap.
- **Local Mac**: an ONNX download and a path change. Low effort.

### 2. Open-vocabulary detection (YOLO-World, Grounding DINO)
Detect from a text prompt, so the class list becomes whatever the standard names.

- **Gains**: the natural fit for Foreman's premise. A spoken standard already
  names its objects, so the parser could hand them straight to the detector and
  "pen" would become groundable like any other noun.
- **Costs**: markedly heavier. Grounding DINO is a transformer at roughly an
  order of magnitude more compute than YOLO26n; YOLO-World is lighter but still
  carries a text encoder. Both need text embeddings computed per standard
  (cacheable - the standard changes rarely). Accuracy on small objects in poor
  light is not obviously better than a specialised detector.
- **Modalix**: the hard one. A text-conditioned detector means compiling a text
  encoder and a fused head to the MLA, and the per-frame budget is currently
  6.23 ms. This is a research effort, not an integration.
- **Local Mac**: feasible today via ONNX or MLX at maybe 50-150 ms/frame, which
  would drop the live view well below 15 fps. Usable for manual inspection,
  not for continuous detection.

### 3. VLM-only semantics for unsupported objects - **what Foreman already does**
The detector grounds what it has a class for; the vision-language model judges
the rest; the console says which is which.

- **Gains**: no new model, no new latency, already shipped and already truthful.
  A standard naming a pen returns `unsupported: ["pen"]`, the console shows
  "pen — not detectable", and the policy records that the part carries no
  detector evidence.
- **Costs**: no detector grounding for those objects, so the anti-hallucination
  guarantee does not extend to them. The VLM can still be wrong about a pen and
  nothing measured will contradict it.

### Recommendation

Keep option 3 as the baseline - it is honest and costs nothing. **Option 1 is the
best next step** if broader grounding is wanted: it preserves the architecture,
the API and the latency budget, and on the local backend it is close to a
drop-in. Option 2 is the most interesting long-term fit for a spoken-standard
system, but on Modalix it is a compilation research project rather than a
feature, and it would not pay for itself before the detector budget is blown.

Whichever is chosen, the rule stands: **detector evidence is only ever claimed
for classes the detector actually has.**
