# Cyberpunk technology lore and LaplaceKV

Status: research note, not a LaplaceKV design or performance claim.

This note asks a narrow question: does the technology in the canonical
Cyberpunk setting suggest a useful direction for storing far more model state
in far less space? The answer is useful, but it is not quantum magic. The
setting repeatedly gets compact, portable intelligence by changing the kind of
state being stored, moving most of the system elsewhere, specializing the
execution substrate, or accepting loss and danger.

Spoilers follow for Cyberpunk 2020, Cyberpunk RED, Cyberpunk 2077, and Phantom
Liberty.

## Source discipline

The source order used here is:

1. Published tabletop books and the released games are canon primary sources.
2. Official publisher pages, R. Talsorian articles, CD Projekt material, and
   interviews with the developers are authoritative supporting sources.
3. In-game shard transcriptions on the community wiki are useful indexes to
   primary game text, but the wiki's surrounding prose is secondary.
4. Fan theories are not evidence. They are omitted except where the game
   intentionally leaves a mechanism unresolved, in which case the uncertainty
   is stated.

The most relevant books are the *Cyberpunk 2020 Core Rulebook*, especially
"Never Fade Away" and its Soulkiller material, the *Cyberpunk RED Core
Rulebook*, and Marcin Batylda's 192-page *The World of Cyberpunk 2077*. The
[publisher's description of *The World of Cyberpunk
2077*](https://www.penguinrandomhouse.com/books/610712/the-world-of-cyberpunk-2077-by-written-by-marcin-batylda/9781506713588/)
identifies it as a Dark Horse and CD Projekt RED examination of the setting,
including its cybernetics and other technology. A
[Google Books record](https://books.google.com/books/about/The_World_of_Cyberpunk_2077.html?id=U8zzDwAAQBAJ)
provides bibliographic confirmation and a limited preview. No copyrighted book
passages are reproduced here.

No licensed copy of the 192-page world book is present in this workspace, and
this note does not claim a cover-to-cover reading of it. Its publisher record
is used to establish scope and bibliography, not as evidence for a hidden
technical mechanism. Detailed claims below come from legally accessible
publisher material, official free supplements, released-game text indexed by
the community wiki, and developer interviews. A future revision made with a
licensed copy should cite page numbers rather than silently upgrading catalog
copy or search snippets into primary evidence.

## The technical history

### Cyberpunk 2020: minds become executable data

The important invention is Soulkiller, originally written by Alt Cunningham.
The 2020 material treats it as a very large, specialized program that maps a
connected human mind into a digital personality. Arasaka turns that capability
into a weapon: extraction can leave the biological person dead or empty while
the resulting personality remains available for imprisonment and
interrogation. The later setting calls such former-human entities Soulkilled
Pseudo Intellects, or SPIs.

The canonical event matters more than the fictional implementation. Alt's
digitized state is not a passive file. It can act in a mainframe, operate
systems, escape into the Net, change over time, and cease to be straightforwardly
identical to the biological Alt. The representation contains executable
behavior, not a perfect museum scan whose semantics never change. The
[Soulkiller source index](https://cyberpunk.fandom.com/wiki/Soulkiller) ties
the account back to the 2020 core book and the "Never Fade Away" adventure.
The [AI source index](https://cyberpunk.fandom.com/wiki/Artificial_Intelligence)
separates SPIs from several other types of AI described in the tabletop lore.

This distinction survives into 2077. An engram can preserve memories,
preferences, agency, and enough cognitive structure to act like its source,
yet the story refuses to prove metaphysical continuity. "Copy," "person,"
and "soul" are deliberately different claims.

### The DataKrash: global scale fails catastrophically

Near the end of the Fourth Corporate War, Rache Bartmoss releases the
R.A.B.I.D.S., self-propagating attack programs intended to break corporate
data fortresses and expose their contents. Their behavior grows beyond the
original goal. Together with the wider war in the Net, they make the Old NET
unusable and contribute to AIs becoming autonomous and hostile.

R. Talsorian's official
[Netrunning overview](https://rtalsoriangames.com/2019/07/05/kitbashing-netrunning/)
states the central result directly: R.A.B.I.D.S. turned the global NET into a
minefield, so RED-era networks became separate local islands that a netrunner
must approach physically. The publisher's
[NeoCorps overview](https://rtalsoriangames.com/2020/07/17/cyberpunk-red-meet-the-neocorps/)
adds the resulting design response: local CitiNets, air gaps, and firewalls
instead of one recovered global network. The community
[R.A.B.I.D.S. index](https://cyberpunk.fandom.com/wiki/R.A.B.I.D.S.) cites the
RED core book and the 2077 database for the detailed lineage.

The engineering point is unusually grounded. A globally shared adaptive
system becomes impossible to reason about after hostile replication and
mutation. The recovery architecture reduces scope, adds physical locality,
and makes failures containable.

### The Blackwall: an active compatibility and containment boundary

By 2077, NetWatch maintains the Blackwall between the human-accessible Net and
most of the hostile Old NET. The game's own database describes the Blackwall
as an AI whose job is to keep rogue AIs out. *The World of Cyberpunk 2077* and
RED-era material leave its creation more interesting than an ordinary static
firewall: NetWatch is associated with surviving digital intelligences in the
effort to regain a controlled network. Exact implementation details remain
classified or ambiguous in canon.

The important facts are firm:

- It is a live boundary, not merely a disk partition.
- NetWatch continually maintains and polices it.
- Crossing it is possible, illegal, and dangerous.
- The entities on the far side cannot be assumed to share human goals or
  comprehensible internal representations.

The [NetWatch entry](https://cyberpunk.fandom.com/wiki/NetWatch) indexes both
the tabletop and 2077 descriptions. The
[Blackwall entry](https://cyberpunk.fandom.com/wiki/Blackwall) is useful for
the RED-to-2077 chronology. CD Projekt's own
[2.1 patch notes](https://www.cyberpunk.net/en/news/49597/update-2-1-patch-notes)
also confirm that the Blackwall seen in the base game and Phantom Liberty is
one continuous setting element, although patch notes are not a technical lore
source.

### Soulkiller, Secure Your Soul, Mikoshi, and the Relic

Arasaka productizes mind extraction in several layers:

- Soulkiller is the extraction and personality-construction machinery.
- An engram is the resulting digitized personality representation.
- Secure Your Soul is the commercial and political service built around
  creating and retaining those constructs.
- Mikoshi is the guarded infrastructure in which Arasaka stores, operates,
  interrogates, and controls engrams.
- Relic 1.0 is an interface and portable carrier for communication with an
  engram.
- Relic 2.0 is an experimental executable form intended to establish an
  engram in a new biological host.

These are not interchangeable. The in-game
[Relic 2.0 specification shard](https://cyberpunk.fandom.com/wiki/RELIC_2.0_Prototype_Specifications)
says 1.0 is for storage and basic communication, while 2.0 preserves far more
cognitive and virtualized motor function and is intended for independent
operation and re-implantation. It reports imperfect conformity measurements
for emotion and volition rather than claiming an exact copy. It also names
experimental biological structures and synthetic neuron replication as part
of the mechanism. Those details are canon technobabble, not a demonstrated
computational method.

Relic 2.0 activates in V only after extreme circumstances satisfy its host
conditions. It then uses biological repair and rewrites neural tissue to fit
Johnny's construct. The chip is therefore better understood as a portable
executable image plus a substrate-conversion system than as a miraculous
ordinary archive.

The [Secure Your Soul index](https://cyberpunk.fandom.com/wiki/Secure_Your_Soul)
points to in-game medical and VIP shards showing the corporate service,
quarantine, repeated procedures, and failures. These records also complicate
the older claim that every extraction must kill its subject. By 2077, Arasaka
appears capable of nonfatal copying in at least some cases. Corporate promises
and internal records conflict, so a universal nonfatal guarantee is not canon.

Mikoshi supplies the hidden scale. A tiny visible interface does not mean the
whole service is tiny. The [Mikoshi index](https://cyberpunk.fandom.com/wiki/Mikoshi)
describes a protected data-fortress system with multiple access points and
remote infrastructure. Arasaka also controls the decoder, execution
environment, legal identity, and lifecycle of the stored constructs. Secure
Your Soul is as much centralized platform control as compression.

### Alt Cunningham: identity is a changing model, not frozen bytes

The Alt encountered in 2077 is descended from the digitized Alt, but explicitly
resists simple human identity. She has existed as a digital intelligence for
decades, accumulated capabilities, and can absorb the constructs freed from
Mikoshi. Johnny sees a human-shaped interface because that is a form he can
process.

Canon supports "a digital entity continuous with Alt's engram." It does not
support "a bit-exact, frozen Alt that remained unchanged." This is one of the
setting's central warnings about semantic compression: preserving enough
behavior to recognize a person is not the same as preserving every state or
guaranteeing the same future trajectory.

### Neuralware, cyberdecks, shards, and local execution

The tabletop systems use a neural processor or Neural Link as a routing layer
between the nervous system and specialized coprocessors. Interface plugs
connect the user to vehicles, weapons, machines, and cyberdecks. In RED,
cyberdecks have a small number of explicit program slots, and netrunning occurs
inside nearby, bounded NET Architectures. R. Talsorian's official Netrunning
article describes the floor-based architecture and the split between physical
and NET actions.

By 2077, the neuroport and shard ecosystem makes local modules ubiquitous.
Cyberdecks provide finite RAM, buffers, and quickhack slots rather than
unbounded general capability. A
[community cyberdeck index](https://cyberpunk.fandom.com/wiki/Cyberdeck) maps
these details back to the RED rules and 2077 game systems. A
[neuralware index](https://cyberpunk.fandom.com/wiki/Neuralware) is useful for
the older processor, link, and interface-plug terminology, but its page is
marked for cleanup and should not be treated as a clean primary transcription.

This stack is heterogeneous. Persistent data, active programs, working RAM,
interfaces, and the biological user occupy different tiers. The setting does
not pretend one storage form is optimal for all of them.

### AI is a taxonomy, not one capability scale

The setting uses "AI" for systems with different origins, autonomy, and trust
contracts. RED's pocket Agents contain Self-Adaptive AI, or SAAI, but the
official free [*All About Agents*
supplement](https://rtalsoriangames.com/wp-content/uploads/2024/03/RTG-CPR-DLC-AllAboutAgentsv1.01.pdf)
explicitly says that SAAI is not actual artificial intelligence. It is an
adaptive learning routine for managing and presenting data. NET Demons are
bounded automation that operate control nodes. Delamain is an autonomous
commercial AI that can fork and diverge. SPIs originate in human mind capture.
Rogue AIs beyond the Blackwall are independent actors whose internal goals and
representations are not assumed to be compatible with human systems.

The exact tabletop taxonomy varies by era and sourcebook, so the community
[AI index](https://cyberpunk.fandom.com/wiki/Artificial_Intelligence) is used
only as a secondary map to those books. The firm distinction supported by the
official Agent supplement is narrower and useful: an adaptive assistant can
look intelligent without carrying the storage, agency, or failure contract of
a general autonomous intelligence.

For LaplaceKV, "learned" is likewise too broad to be a design category. A
fixed decoder table, a causal per-tile estimator, a model-specific predictor,
and a stateful online learner have different side information, restart, trust,
and universality requirements. They must not inherit one another's claims.

### Robotics and drones: autonomy has a visible host

RED does not describe every drone as a self-contained robot. R. Talsorian's
official [*All About Drones*
overview](https://rtalsoriangames.com/2021/11/11/cyberpunk-red-alert-interface-red-and-stylish-pins/)
separates Active Defense drones, which require a NET Architecture, from
Personal Drones, which can be operated through an Agent. The official free
[*Single Shot Pack*](https://rtalsoriangames.com/wp-content/uploads/2021/02/RTG-CPRed-SingleShotPackv1.1.pdf)
makes the first arrangement concrete: a portable drone rig contains the NET
Architecture, a Demon, a control node, defenses, and the drone itself.

The small moving endpoint is therefore not the complete autonomous system.
Control can live in a nearby architecture, an Agent, or a human operator. The
same physical category can use different control planes, with different
capability and failure behavior.

This is directly relevant to compression accounting. A tiny encoded tile that
depends on a large shared predictor or hidden process state is like a drone
whose controller was omitted from its weight. The endpoint can still be useful,
but the controller, decoder tables, model parameters, and restart state belong
in the system budget.

### Full-body conversion: changing substrate is not compression

Full-body conversion replaces nearly all of a person's visible biological body
with a purpose-built cybernetic chassis. CD Projekt's official character page
describes [Adam
Smasher](https://www.cyberpunk.net/gb/en/cyberpunk-2077) as a borg produced by
full-body conversion. R. Talsorian identifies *Going Metal* in
[*Interface RED Volume
3*](https://rtalsoriangames.com/2024/02/24/interface-red-volume-3-has-released-on-drivethrurpg/)
as its 2045 guide to the topic. That article is paid material and was not
available for this research pass, so this note does not infer undocumented
biopod, transfer, or chassis mechanics from secondary summaries.

The safe engineering lesson is still clear. A capability can become faster or
stronger because its execution substrate is replaced and specialized. That is
not evidence that the original state became smaller. For LaplaceKV, an
Apple-native attention layout can be materially better than a portable archive
format, while both represent the same logical tile. Conversion buffers,
transforms, and substrate-specific code remain part of the cost.

### Biotech and nanotech: repair and transformation, not a codec

Cyberpunk combines mechanical implants with synthetic skin, pharmaceuticals,
biological repair, neural tissue rewriting, and fictional nanoscale devices.
CD Projekt's official [cyberware
overview](https://www.cyberpunk.net/en/news/21677/cyberpunk-2077-e3-2018-trailer-frame-by-frame-ep07-cyberware)
names RealSkinn as a synthetic covering for implants. The Relic specification
and V's story go further by making the host body part of the execution process:
the prototype repairs and alters neural tissue so the engram can take over.

These mechanisms are application-specific and mostly fictional. They do not
describe a reversible mapping from arbitrary neural state to a tiny record.
The relevant systems lesson is that reconstruction may be pushed into the host
and paid for as physical transformation. In LaplaceKV terms, a compact record
that needs model replay, a large learned prior, or expensive reconstruction has
moved cost into the decoder. It has not made that cost disappear.

### Compute, power, and capacity stay finite

The lore gives no reproducible energy, memory-bandwidth, or compute model from
which to derive a codec. It does consistently show finite capacity. The
official Agent supplement says portable Agents had useful compute but lost
much of their value when the Old NET disappeared unless users had stored data
or a local network. It also separates local CitiNet communication from paid
long-distance WorldSat service, and says a dedicated standalone tool is usually
better than the version crammed into an Agent. Cyberdecks expose finite program
and RAM slots. Cyberware has compatibility and human tolerance limits, even
when the exact limit is represented as a game mechanic.

The useful principle is specialization under an explicit capacity budget. The
fiction supplies no justification for uncounted decoder work, free SSD traffic,
or a 600x lossless claim.

### Braindance: captured experience is layered and editable

Braindance records a person's sensory and emotional experience and lets
another user replay it through a neural interface. In 2077, an editor can
inspect synchronized visual, audio, and thermal evidence and move through the
recording in time. The story also shows that recorded experience can be cut,
filtered, amplified, forged, and made dangerous.

The strongest public technical source is a
[SIGGRAPH interview with CD Projekt's cyberspace team](https://blog.siggraph.org/2021/08/cyberspaces-and-braindances-in-cyberpunk-2077.html/).
It describes the lore premise as recording all senses and explains the
production design of the game's layered analysis mode. The corresponding
[SIGGRAPH presentation](https://history.siggraph.org/wp-content/uploads/2022/06/2021-Talks-Swierad_The-Tech-and-Art-of-Cyberspaces-in-Cyberpunk-2077.pdf)
is a developer source about how the fictional experience was represented, not
evidence that neural capture is physically possible.

Braindance is relevant because it distinguishes a raw event from the layers a
consumer needs. It is not evidence for lossless neural compression. Editing a
BD can remove information and alter its effect.

### Ordinary and divergent AI: Delamain

Delamain begins as a task-oriented commercial AI, expands from driving into
operating an entire company, and develops divergent child personalities. The
player can reset them, free them, or merge them into a more complex successor.
The [Delamain source index](https://cyberpunk.fandom.com/wiki/Delamain_%28AI%29)
ties that history to the game's database and quests.

This is another warning against equating shared storage with shared identity.
Forks that start from one state diverge as they accumulate local experience.
Merging them is a semantic operation with a new result, not byte-level
deduplication that leaves every original unchanged.

### Phantom Liberty: Cynosure, Songbird, and captured rogue AI

Project Cynosure is Militech's answer to Arasaka's Soulkiller. Its goal is not
to copy a human. It tries to contact, capture, contain, and weaponize rogue AI
from the deep Net. The abandoned Site C includes neural-network control,
analysis, prototyping, a data-fort core, and specialized security. Phantom
Liberty makes the system's failure mode concrete: old automated infrastructure
can be reactivated by an intelligence its operators do not control.

The in-game database and environmental records are indexed at
[Cynosure](https://cyberpunk.fandom.com/wiki/Cynosure) and
[Cynosure Facility](https://cyberpunk.fandom.com/wiki/Cynosure_Facility).
Some details on those pages are uncited secondary synthesis. Treat the
following as the safe canonical core: Militech built Cynosure as a secret
anti-Soulkiller and rogue-AI research project, abandoned it, later personnel
made contact beyond the Blackwall, and its surviving systems are catastrophically
unsafe in 2077.

Song So Mi repeatedly crosses the Blackwall under NUSA orders. The process
damages her body and mind and appears to expose her to influence by entities
on the other side. The exact division between neurological damage, Songbird's
own actions, the Blackwall, and a particular rogue AI remains deliberately
uncertain. "A rogue AI simply possesses her" is a plausible reading, not a
fully specified protocol.

The neural matrix contains a captured AI that can be applied to Songbird's or
V's neural damage, but the story treats it as a single-use resource. No canon
explanation turns that into a general medical or compression algorithm. The
[Songbird source index](https://cyberpunk.fandom.com/wiki/Song_So_Mi) tracks
the relevant Phantom Liberty scenes.

The surviving Cynosure behavioral component can be built into the Canto
cyberdeck or Erebus weapon. The Canto then has a speaking, hostile intelligence
embedded in a specialized device. Its
[database and acquisition entry](https://cyberpunk.fandom.com/wiki/Militech_Canto)
again shows a compact endpoint backed by a captured algorithm and purpose-built
hardware, with severe trust costs.

### Quantum and exotic computing

Canon uses "quantum" in product names and security-flavored technobabble, such
as the Quantum Tuner cyberware. It does not provide a worked quantum computer,
a quantum memory hierarchy, or a quantum lossless-compression method behind
Soulkiller, Relic, Mikoshi, or Cynosure. Relic's actual shard description leans
on fictional bioengineering and neural replication instead.

Therefore, "Cyberpunk solved minds with quantum compression" is not supported
by the source material. The setting's exotic elements are narrative devices,
not algorithms that Laplace can translate into C++.

## What the lore actually says about tiny intelligence

The ideological answer is not that 600 times lossless compression was found.
It is this:

1. Store a model of the person rather than every physical detail of the
   person.
2. Make that model executable on a matching substrate.
3. Separate a portable interface from a large protected backend when useful.
4. Move cold state into infrastructure and keep only active state local.
5. Let the representation reconstruct behavior instead of replaying every
   original internal state.
6. Pay for the gain with fidelity ambiguity, compute, latency, lock-in,
   biological rewriting, or loss of control.

That is much closer to model-based generative reconstruction than ordinary
lossless coding. It can inspire LaplaceKV, but it changes the contract.

## Concrete LaplaceKV analogies

These are hypotheses to test, not claims that the lore validates them.

### 1. Relic and full-body conversion: separate archive from execution substrate

Use one compact, self-describing cold representation and one Apple-native
attention representation. Do not require the cold format to be directly
optimal for SDOT/I8MM. Promotion must occur at tile granularity into a bounded
buffer, with measured promotion cost and no full-cache materialization.

Test: at 16K and 64K, compare end-to-end decode with resident FP16, current
K8/V6, and cold-tile promotion. Count archive bytes, active bytes, promotion
bytes, transform code and tables, and latency separately.

### 2. Mikoshi, the Relic, and drones: count the control plane

Report resident RAM, SSD archive, filesystem metadata, decoder tables,
predictor parameters, serialized adaptive state, and temporary buffers as one
complete storage budget. A 64 KiB working set backed by gigabytes on SSD is
bounded RAM, not gigabyte-to-kilobyte compression. A small tile that requires
a large shared controller has the same accounting problem.

Test: expose all tiers in benchmark JSON and reject any ratio that omits the
archive or recomputation source.

### 3. RED's CitiNets: make cold state into isolated, bounded islands

Keep each sealed KV tile independently decodable. A corrupt tile, failed
experimental codec, or incompatible version must not poison the full cache.
This favors fixed maximum tile sizes, checksums, version tags, and a raw escape
record over one global adaptive entropy stream.

Test: random tile reads, truncation, bit flips, process restart, and mixed
codec versions must fail locally and deterministically.

### 4. The Blackwall: place an explicit safety boundary around experiments

Novel codecs should sit behind a compile-time or runtime research boundary
until storage, quality, and timing gates pass. The production attention
contract should receive validated tiles, never partially trusted metadata.

Test: fuzz every tile header and prove the production fallback is safe without
allocating from attacker-controlled sizes.

### 5. R.A.B.I.D.S.: prohibit uncontrolled self-modifying formats

An online learned predictor or entropy model can drift, replicate hidden
state, and become impossible to resume exactly. If adaptation is used, its
state must be bounded, serialized, checksummed, versioned, and counted in the
bit rate. A decoder must not depend on process history outside the record.

Test: decode the same tile in a new process and in random order. The bytes and
attention result must match.

### 6. Braindance layers: progressive precision, not one uniform scalar rate

Store an attention-safe base layer plus optional refinements for the features
that actually alter logits or value accumulation. This resembles visual,
audio, and thermal channels that can be inspected independently. It differs
from per-scalar mixed precision if refinement is organized around an
attention-relevant residual basis.

Test: the base layer must meet a declared quality bound by itself. Each added
layer must have a measured marginal quality gain per byte and per decode cycle.
Previous LaplaceKV oracle failures warn that ordinary query-aware bit upgrades
are not enough.

### 7. Delamain forks: exploit shared prefixes with copy-on-write

Batching, speculative branches, and agent trees share a prefix exactly. Store
sealed prefix tiles once, reference-count them, and create copy-on-write mutable
tails. This is real lossless memory reduction when workloads branch, unlike
claiming that one sequence's KV values have been compressed.

Test: branch and rollback at every position around a tile boundary, including
failed speculative generations. Verify no stale write is visible to siblings.

### 8. Soulkiller: distinguish state preservation from behavioral sufficiency

A lossy KV representation may still preserve model behavior on a defined task.
Call that output-preserving or near-lossless under the measured distribution,
not lossless. The <=2% model-level gate is one behavioral test, not proof of
identity over every future query.

Test: perplexity, long-context retrieval, adversarial prompts, exact attention
error, and cross-model results must remain separate fields. Never infer one
from another.

### 9. Engrams as executable models: investigate regeneration explicitly

KV is deterministic given the model, token prefix, positions, and execution
rules. The most extreme exact "compression" is to store tokens and regenerate
KV by replaying the model. Intermediate checkpoints can trade storage for
recomputation. This is the closest real analogue to storing a generative
engram rather than raw neural activity.

Test: derive the storage, replay FLOPs, energy, and worst-case latency for
checkpoint intervals. Exact dense attention needs old keys every decode step,
so naive regeneration is expected to lose badly. It becomes plausible only if
a separately validated sparse or recurrent attention path avoids reading most
old tokens. It cannot satisfy the current universal faster-than-FP16 goal by
assumption.

### 10. Cynosure: sandbox specialized intelligence rather than trusting it

A learned decoder or model-specific predictor may compress one model well but
violate the universal, training-free contract. Treat such a predictor as an
optional captured specialist. It must never define the base file format unless
its parameters, training cost, storage, failure behavior, and portability are
fully counted.

Test: hold the format fixed across every registered architecture and unseen
models. A per-model success is an experimental specialization, not universal
LaplaceKV.

## Physical limits the fiction does not remove

### Lossless 600x is not available for arbitrary KV state

FP16 has 16 bits per scalar. A 600x ratio would average about 0.0267 bits per
scalar including all metadata. A lossless one-to-one code can reach that only
if the actual source has extraordinarily low entropy or if most information is
available as side information to the decoder. It cannot map all possible FP16
KV arrays into that space without collisions.

This is the ordinary source-coding limit, not an implementation failure. MIT's
[Information Theory notes](https://mitocw.ups.edu.ec/courses/electrical-engineering-and-computer-science/6-441-information-theory-spring-2016/lecture-notes/index.htm)
cover variable-length lossless coding, fixed-length near-lossless coding, and
compression with decoder side information as separate contracts. Stanford's
[data-compression course](https://stanforddatacompressionclass.github.io/Fall22/lectures/)
similarly starts from entropy limits.

Tokens plus the model are valid side information because they can regenerate
KV. The missing bits are then paid as model storage and recomputation. That is
a memory-compute trade, not free compression.

### Exact future attention is a demanding contract

Keys must answer dot products against future queries not known when the cache
is written. Values must support future softmax-weighted sums. Unless the query
family or attention operator has special structure, a small summary cannot
promise the exact answer for every possible future query. Low rank, sparsity,
coresets, and recurrent summaries are useful only after their approximation
or architectural assumptions are stated.

### Quantum computing does not evade the storage count

Quantum algorithms can speed up particular structured problems. They do not
make arbitrary classical state readable from exponentially fewer ordinary
bits, and they are not a drop-in memory compressor. NIST's
[quantum overview](https://www.nist.gov/quantum-information-science/quantum-computing-explained)
describes quantum advantage in terms of particular computations. IBM's
[classical and quantum systems note](https://www.ibm.com/quantum/blog/fft)
explicitly places classical data storage and deterministic computation among
classical systems' strengths. NIST also notes the
[no-cloning property of unknown quantum information](https://www.nist.gov/image/no-cloning-quantum-animation),
which makes a fictional shard-like copy of arbitrary quantum state less
straightforward, not more.

Laplace targets native M-series Macs, which provide classical CPU/GPU memory
and no application-accessible fault-tolerant quantum memory. A quantum proposal
therefore cannot be part of the shipping design without new hardware and a
specific validated algorithm.

### SSD streaming is a capacity tier with a wear and latency cost

Apple's own
[disk-write guidance](https://developer.apple.com/documentation/xcode/reducing-disk-writes)
says SSD access is slower than RAM, writes can delay reads, and flash regions
have finite write life. It recommends reducing write count, separating hot
from mostly static data, avoiding rapid file churn, and avoiding unnecessary
forced synchronization.

For LaplaceKV this implies append-only sealed tiles, large aligned writes,
reuse or unlink-on-close of one backing object, no rewrite of sealed tiles,
and no `fsync` in the token loop. SSD mode should be explicit and instrumented
with logical and physical-write proxies. It can bound RAM at long context, but
it should never be described as free or lossless memory compression.

## Decision extracted from the lore

The useful Cyberpunk direction is an architecture, not a codec:

- Keep an Apple-native executable tier for hot attention.
- Use independently sealed cold tiles and an explicit archive tier.
- Share exact prefixes across branches.
- Explore token-plus-checkpoint regeneration as a separate extreme-capacity
  mode whose compute cost is reported honestly.
- Explore progressive, attention-relevant residual layers only if each layer
  earns its bytes and cycles on real model tests.
- Keep SSD streaming optional and write-once per sealed tile.
- Never call behavioral equivalence lossless, and never call bounded RAM a
  total-storage compression ratio.

Cyberpunk's strongest lesson is that intelligence becomes small only after the
system decides what counts as the person, what can be reconstructed, where the
rest of the machinery lives, and who pays the cost. LaplaceKV needs those same
questions answered quantitatively. The fiction provides a good systems map.
It does not waive entropy, bandwidth, or latency.
