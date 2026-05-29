<h1 align="center">CoderMind</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.ja-JP.md">日本語</a> |
  <a href="README.ko-KR.md">한국어</a> |
  <a href="README.hi-IN.md">हिन्दी</a>
</p>

## कोडिंग एजेंट्स को संपादन से पहले प्लानिंग करने दें

कोडिंग एजेंट्स लोकल संपादन में मजबूत होते हैं, लेकिन एक स्थिर प्लानिंग संरचना के बिना रिपॉज़िटरी-स्तर के कार्य अक्सर विफल हो जाते हैं। आवश्यकताएँ बहक जाती हैं, आर्किटेक्चर के निर्णय खो जाते हैं, मल्टी-फ़ाइल जनरेशन असंगत हो जाती है, और अपडेट छिपी हुई dependencies को मिस कर सकते हैं।

CoderMind, Claude Code और GitHub Copilot को रिपॉज़िटरी-स्तर कोडिंग के लिए एक **persistent RPG workspace** देता है। यह वर्कस्पेस एक **Repository Planning Graph (RPG)** के चारों ओर बना है, जो आवश्यकताओं, features, आर्किटेक्चर, फ़ाइलों, कोड entities और dependencies को जोड़ता है।

CoderMind के साथ, एजेंट्स ग्राफ-संचालित वर्कफ़्लो के माध्यम से काम करते हैं:

- **Build (निर्माण)**: आवश्यकताओं को RPG प्लान में बदलें, फिर एक मल्टी-फ़ाइल रिपॉज़िटरी बनाएँ।
- **Understand (समझें)**: किसी मौजूदा रिपॉज़िटरी को RPG में मैप करें, फिर खोजें, अन्वेषण करें और समझाएँ।
- **Update (अपडेट करें)**: प्रभावित RPG नोड्स को पहचानें, संपादन प्लान बनाएँ, और कोड व ग्राफ को एक साथ अपडेट करें।

### अपना वर्कफ़्लो चुनें

| लक्ष्य | वर्कफ़्लो | यहाँ से शुरू करें |
|---|---|---|
| आवश्यकताओं से एक नई रिपॉज़िटरी बनाना | Build वर्कफ़्लो (requirements → RPG → code) | [`Quick Start: नई रिपॉज़िटरी`](#quick-start-नई-रिपॉज़िटरी) |
| किसी मौजूदा रिपॉज़िटरी को समझना | Understand वर्कफ़्लो (repository → RPG → search/explore) | [`Quick Start: मौजूदा रिपॉज़िटरी`](#quick-start-मौजूदा-रिपॉज़िटरी) |
| किसी मौजूदा रिपॉज़िटरी को अपडेट करना | Update वर्कफ़्लो (change request → affected RPG nodes → edit plan → code/RPG update) | [`Quick Start: मौजूदा रिपॉज़िटरी`](#quick-start-मौजूदा-रिपॉज़िटरी) |

### विस्तृत पाइपलाइन

नए उपयोगकर्ता इस सेक्शन को छोड़कर सीधे नीचे दिए गए Quick Start से शुरू कर सकते हैं।

<details>
<summary>कमांड-स्तर का पूर्ण वर्कफ़्लो आरेख</summary>

```text
Forward Direction: Requirements → RPG → Code

 Phase 1: Feature Specification       Phase 2: RPG Construction & Planning                             Phase 3
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ feature  │ │ feature  │ │ feature  │ │  build   │ │  build   │ │ design   │ │ design   │ │  plan    │ │          │
│  _spec   ├─▶  _build  ├─▶_refactor ├─▶ skeleton ├─▶  data    ├─▶  base    ├─▶interfaces├─▶  tasks  ├─▶ code_gen │
│          │ │          │ │          │ │          │ │  flow    │ │ classes  │ │          │ │          │ │   (TDD)  │
└──────────┘ └──────────┘ └────┬─────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────┬─────┘
 feature_     feature_        │        skeleton     data_flow    base_        interfaces   tasks        source
 spec/        build           │        .json        .json        classes      .json        .json        code
 feature_     .json           │        skeleton_    data_flow    .json
 spec.json                    │        summary.txt  _viz.html
                              │
                       ┌──────▼──────┐
                       │ feature_edit│ optional pre-planning edits to feature_tree.json
                       └─────────────┘
                                        ╰───── rpg.json (created → progressively enriched) ─────╯
                                                                            │
                                                                            ▼
                                                                     ┌──────────┐
Surgical edit workflow: Requirements -> RPG update -> Code Update    │ rpg_edit │ optional synchronized RPG + code + dep_graph edits
                                                                     └──▲────▲──┘
                                                                        │    │
Reverse Direction: Code → RPG                                           │    │
                                                                        │    │
┌──────────────────┐         ┌──────────┐       ┌──────────┐            │    │
│ Existing Codebase│────────▶│  encode  │──────▶│update_rpg│────────────┘    │
│                  │         │  (full)  │       │ (manual  │                 │
└──────────────────┘         └────┬─────┘       │ fallback)│                 │
                              rpg.json          └──────────┘                 │
                              dep_graph.json     rpg.json / dep_graph.json   │
                                  │                                          │
                                  └──────────────────────────────────────────┘
                                                  ▲
                                                  │ post-commit hook normally runs incremental updates

MCP Server: search_rpg / explore_rpg / get_node_detail / list_rpg_tree
```

</details>

### CoderMind वास्तविक उपयोग में

नीचे दी गई छवि इस रिपॉज़िटरी के लिए जनरेट किए गए ग्राफ़ विज़ुअलाइज़ेशन का एक भाग है। `/cmind.encode` चलाने के बाद, पूर्ण इंटरैक्टिव ग्राफ़ देखने के लिए `<workspace>/.cmind/reports/rpg.html` खोलें। वर्तमान वर्कस्पेस के हल किए गए पथ देखने के लिए `cmind version` चलाएँ।

![CoderMind repository graph visualization](../docs/cmind_visualized_graph.png)

## इंस्टॉलेशन

### पूर्वापेक्षाएँ

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- एक इंस्टॉल और प्रमाणित AI कोडिंग एजेंट CLI: [GitHub Copilot](https://docs.github.com/en/copilot) या [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)

### CoderMind इंस्टॉल करें

```bash
# Persistent इंस्टॉलेशन (अनुशंसित)
uv tool install cmind-cli --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind"
cmind check

# एक बार के उपयोग के लिए
uvx --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" cmind init <project-name>
```

`0.1.3` से, wheel pipeline scripts और slash-command templates को packaged assets के रूप में शामिल करता है, इसलिए `cmind init` ऑफ़लाइन वातावरणों (जैसे air-gapped या corporate proxy वातावरण) में भी काम करता है।

## Quick Start: नई रिपॉज़िटरी

जब आप CoderMind से आवश्यकताओं को एक नए कोडबेस में बदलवाना चाहते हैं, तब इस मार्ग का उपयोग करें।

> [!WARNING]
> बहुत अधिक जनरेटेड कोड वाली परियोजनाओं के लिए, `/cmind.design_interfaces` और `/cmind.code_gen` को चलने में काफ़ी समय लग सकता है। उदाहरण: 100 features में लगभग 30 मिनट लगते हैं।

1. नई परियोजना को आरंभीकृत करें:

   ```bash
   cmind init my-project
   cd my-project
   ```

   सामान्य विकल्प:

   ```bash
   cmind init my-project --ai claude --script sh
   cmind init my-project --ai copilot
   ```

2. **[वैकल्पिक]** अपने आवश्यकता दस्तावेज़ `my-project/docs/` में रखें।

3. परियोजना निर्देशिका में अपना AI कोडिंग एजेंट लॉन्च करें।

4. फॉरवर्ड पाइपलाइन चलाएँ:

   ```text
   /cmind.feature_construct <feature description>
   [Optional] /cmind.feature_edit <edit instructions>
   /cmind.plan
   /cmind.code_gen
   [Optional] /cmind.rpg_edit <edit instructions>
   ```

> [!IMPORTANT]
> **हर Coding Agent का इनवोकेशन थोड़ा अलग होता है**:
>
> - **Claude Code**: चैट में सीधे `/cmind.feature_construct ...` टाइप करें — slash command पहचाने जाते हैं और संबंधित workflow ट्रिगर हो जाता है।
> - **GitHub Copilot CLI**: slash command समर्थित नहीं हैं (कस्टम agent समर्थित हैं), इसलिए पहले `/agent cmind.feature_construct` से लक्ष्य agent पर स्विच करें, फिर `start` टाइप करके इसका अंतर्निहित workflow चलाएँ।

CoderMind क्रमिक रूप से `~/.cmind/workspaces/<workspace-id>/data/rpg.json` बनाता है और इसका उपयोग आवश्यकताओं, प्लानिंग आउटपुट, जनरेटेड कोड और dependency जानकारी को संरेखित रखने के लिए करता है। आपके वर्कस्पेस की स्रोत फ़़ाइलें दूषित नहीं होंगी।

## Quick Start: मौजूदा रिपॉज़िटरी

जब आपके पास पहले से एक रिपॉज़िटरी है और आप चाहते हैं कि AI एजेंट इसे RPG कॉन्टेक्स्ट के साथ समझे या संपादित करे, तब इस मार्ग का उपयोग करें।

> [!WARNING]
> बड़ी परियोजनाओं के लिए, `cmind init . --encode` और `/cmind.encode` को चलने में काफ़ी समय लग सकता है। उदाहरण: 200 स्रोत फ़ाइलों में लगभग 100 मिनट लगते हैं।

1. रिपॉज़िटरी रूट में CoderMind को आरंभीकृत करें और प्रारंभिक ग्राफ़ बनाएँ:

   ```bash
   cd existing-repo/
   cmind init . --encode    # --encode वर्तमान कोड से RPG उत्पन्न करता है
   ```

   यदि आप गैर-खाली निर्देशिका के लिए पुष्टि संकेत को छोड़ना चाहते हैं:

   ```bash
   cmind init . --force --encode
   ```

2. रिपॉज़िटरी में अपना AI कोडिंग एजेंट लॉन्च करें।

3. **[वैकल्पिक]** MCP टूल्स और स्लैश कमांड्स के माध्यम से जनरेटेड RPG का उपयोग करें। नीचे दिए गए कमांड केवल मैन्युअल रूप से चलाने पर आवश्यक हैं:

   ```text
   /cmind.encode                                  # आवश्यकता पड़ने पर पूर्ण RPG को पुनर्निर्मित करें
   /cmind.update_rpg                              # मैन्युअल वृद्धिशील अपडेट (fallback)
   /cmind.rpg_edit <edit instructions>            # ग्राफ़-जागरूक कोड संपादन
   ```

4. हर commit के बाद, CoderMind द्वारा इंस्टॉल किया गया git hook स्वचालित रूप से `cmind hook <name>` dispatcher को कॉल करता है, RPG को अपडेट करता है और उसे कोड परिवर्तनों के साथ संरेखित रखता है। यदि hook विफल हो जाता है या छोड़ दिया जाता है, तो `/cmind.update_rpg` मैन्युअल रूप से चलाएँ।

## `cmind init` के बाद क्या होता है

`cmind init` आपकी स्रोत फ़़ाइलों को संशोधित नहीं करता है, **और आपके वर्कस्पेस में रनटाइम स्टेट नहीं लिखता है**। यह आपके वर्कस्पेस में केवल command definitions, MCP कॉन्फ़़िगरेशन और hooks जोड़ता है। CoderMind का रनटाइम डेटा (outputs और logs) home-side निर्देशिका `~/.cmind/workspaces/<workspace-id>/` के अंतर्गत रखा जाता है, जहाँ `<workspace-id>` वर्कस्पेस के absolute path से जनित एक पठनीय slug है (उदाहरण: `home-hys-projects-myrepo`)।

```text
my-project/
├── docs/                 # /cmind.feature_construct के लिए वैकल्पिक आवश्यकता दस्तावेज़
├── .github/ or .claude/  # Coding Agent कमांड परिभाषाएँ और सेटिंग्स
├── .vscode/              # लागू होने पर Copilot/VS Code MCP कॉन्फ़िगरेशन
├── .cmind/              # जनरेटेड रिपोर्ट और कॉन्फ़िगरेशन फ़ाइलें
└── .git/hooks/           # cmind init द्वारा इंस्टॉल किए गए post-commit / post-merge (प्रत्येक hook केवल एक पंक्ति: `cmind hook <name>`)
```

पूर्ण लेआउट और डेटा फ़ाइल संदर्भ के लिए [docs/project-structure.md](docs/project-structure.md) देखें।

## CoderMind अपडेट करें

```bash
uv tool install cmind-cli \
   --from "git+https://github.com/microsoft/RPG-ZeroRepo.git#subdirectory=CoderMind" \
   --force \
   --reinstall

# किसी मौजूदा वर्कस्पेस को अपडेट करें
cd <your-workspace>
cmind update
```

## समर्थित प्लेटफ़ॉर्म्स

**Coding Agent समर्थन**:

| Agent          | CLI उपयोग | VS Code एक्सटेंशन उपयोग |
| -------------- | --------- | ----------------------- |
| Claude Code    | ✅        | ✅                      |
| GitHub Copilot | ✅        | ✅                      |
| Codex          | ⌛        | ⌛                      |

**ऑपरेटिंग सिस्टम समर्थन**:

| ऑपरेटिंग सिस्टम | स्थिति |
| ---------------- | ------ |
| Linux            | ✅     |
| macOS            | ⌛     |
| Windows          | ⌛     |

## दस्तावेज़ीकरण

- [स्लैश कमांड संदर्भ](docs/commands.md) — हर `/cmind.*` कमांड के लिए इनपुट, आउटपुट और उदाहरण।
- [CLI संदर्भ](docs/cli-reference.md) — `cmind init`, `cmind update`, `cmind check`, `cmind version` और सभी विकल्प।
- [कॉन्फ़िगरेशन](docs/configuration.md) — AI सहायक सेटअप, MCP पंजीकरण, hooks, ऑटो-अनुमोदन और समस्या-निवारण।
- [परियोजना संरचना](docs/project-structure.md) — CoderMind द्वारा बनाई गई फ़ाइलें और निर्देशिकाएँ।

## आगामी सुविधाएँ

- **सरल जनरेशन कमांड्स:** वर्तमान बहु-चरण जनरेशन प्रवाह को कम कमांड्स में मर्ज किया जाएगा, जैसे `/cmind.generate_repo` और `/cmind.generate_feature`। `/cmind.plan` 0.1.4 में रिलीज़ हो चुका है।
- **बहु-भाषा समर्थन:** Go, C++, Rust, JavaScript/TypeScript और अन्य के लिए समर्थन जोड़ा जाएगा।
- **अधिक प्लेटफ़ॉर्म एकीकरण:** विभिन्न सिस्टम्स पर विभिन्न AI कोडिंग एजेंट्स के लिए CLI और VS Code एक्सटेंशन वर्कफ़्लो में CoderMind समर्थन।

## समस्या-निवारण

**AI सहायक CLI नहीं मिला:** `cmind check` चलाएँ, चयनित सहायक CLI को इंस्टॉल और प्रमाणित करें, फिर `cmind init` या `cmind update` पुनः चलाएँ।

## लाइसेंस

MIT License — विवरण के लिए [LICENSE](LICENSE) देखें।

## आभार

[GitHub Spec-Kit](https://github.com/github/spec-kit) पर आधारित।
