# Prompt Dissection 

It is incredibly exciting that you are building a hardware design agent. Translating ambiguous human language into deterministic actions is the holy grail of agentic AI.

To understand how models like Gemini or Claude handle this, it helps to know that we don't "parse" text using traditional syntax trees. Instead, our self-attention mechanisms weigh the semantic relationships between words, allowing us to map natural language onto the rigid schemas of the tools we have been provided. We essentially perform continuous **intent extraction**, **entity recognition**, and **constraint mapping**.

Here is a breakdown of how an LLM decomposes your C++ prompt into a set of unambiguous assertions, followed by how you can apply this methodology to your silicon design agent.

---

### 1. Intent and Scope Extraction

Before taking action, the agent establishes the "Why" and "Where."

* **Primary Intent 1 (Refactor):** Modernize the codebase for cross-platform compatibility.
* **Primary Intent 2 (Feature Creation):** Build an analytical tool to demonstrate precision loss across different numerical representations.
* **Scope:** The entire repository (for the refactor) and specifically `./applications/precision/constants/` (for the new feature).

### 2. Decomposing into Unambiguous Assertions (Task Breakdown)

The LLM translates the intent into a directed acyclic graph (DAG) of tasks. Each assertion represents a distinct, verifiable state change or computation.

**Phase 1: Codebase Refactoring**

* **Assertion 1.1:** The agent must identify all files containing the regular expression pattern matching `M_PI`, `M_E`, `M_SQRT2`, etc.
* **Assertion 1.2:** The agent must verify if `#include <numbers>` is present in those files.
* **Assertion 1.3:** The agent must replace `M_<CONSTANT>` with `std::numbers::<constant_lowercase>`.

**Phase 2: Example Scaffolding**

* **Assertion 2.1:** A new directory/file structure must be instantiated at `./applications/precision/constants/main.cpp`.
* **Assertion 2.2:** The file must include necessary headers for standard types, `qd_cascade`, `fixpnt`, `posits`, and `lns`.

**Phase 3: Mathematical Implementation & Constraints**

* **Assertion 3.1:** The agent must instantiate variables for Pi (and potentially others) across the specified types: IEEE 754 half/single/double, `qd_cascade`, `fixpnt`, `posits`, and `lns`.
* **Assertion 3.2:** The agent must calculate the Unit in the Last Place (ULP) for each data type at the value of Pi.
* **Assertion 3.3:** The agent must calculate the relative error constraint. The prompt specifies this must be done using at least quad-double precision. The agent recognizes the need to implement the following mathematical formula in C++:

$$\epsilon_{rel} = \left| \frac{v_{approx} - v_{true}}{v_{true}} \right|$$

* *Constraint verification:* $v_{true}$ must be represented as `qd_cascade` (or `__float128`), and $v_{approx}$ is the casted value of the lower-precision format.

**Phase 4: Output Formatting**

* **Assertion 4.1:** The output must be a printed table.
* **Assertion 4.2:** The table must strictly contain the columns: Number System, ULP, Value, Relative Error.

---

### 3. Mapping Assertions to Tool Calls

For an agent to execute these assertions, you must design tools with highly specific schemas. The LLM acts as the orchestrator, looking at the assertions and selecting the right tool.

Here is how the LLM maps the above assertions to hypothetical tools:

| Assertion Phase | Required Tool Capability | Example Tool Name | Arguments Passed by LLM |
| --- | --- | --- | --- |
| **Phase 1** | Global string/regex search | `search_codebase` | `{"pattern": "M_[A-Z0-9_]+"}` |
| **Phase 1** | AST-aware code editing | `edit_source_file` | `{"filepath": "...", "search": "M_PI", "replace": "std::numbers::pi"}` |
| **Phase 2** | File system manipulation | `create_file` | `{"filepath": "./applications/precision/constants/main.cpp", "content": "..."}` |
| **Phase 3 & 4** | Compilation & execution | `run_build_and_execute` | `{"target_dir": "./applications/precision/constants/", "command": "make run"}` |

---

### 4. Translating this to your Silicon Design Agent

If you are building an agent for Verilog/SystemVerilog, your users will provide similarly complex prompts.

**Example Hardware Prompt:**

> *"Go through the ALU module and replace all asynchronous resets with synchronous active-low resets. Then, create a SystemVerilog testbench in `./dv/` that verifies the pipeline stages don't stall when the multiplier raises a busy flag. Generate a coverage report for the FSM."*

To enable an LLM to decompose this successfully, you need to design tools that map to hardware-specific assertions:

1. **Semantic Search Tool:** A simple text search isn't enough for Verilog. You need a tool like `query_ast` so the LLM can ask: *"Find all `always` blocks in `alu.sv` triggered by `negedge rst_n`."*
2. **RTL Editing Tool:** A tool that safely modifies port lists and sequential logic blocks without breaking linting rules.
3. **Simulation Orchestrator:** A tool named `run_eda_simulator` that takes the testbench path, compiles it, and returns the stdout (pass/fail) and the path to the coverage database.
4. **Coverage Extractor:** A tool that reads the generated coverage database and returns a structured JSON of the FSM state/transition coverage back to the LLM.

The secret to a good agent isn't necessarily having a smarter LLM; it is giving the LLM atomic, highly specific tools so that when it breaks down the prompt into assertions, it has a 1:1 mapping for how to execute them.

