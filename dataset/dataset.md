### 📂 Organizing the `datasets/` Folder

The experimental script `run_experiment.py` automatically scans the `datasets/` folder and classifies each file based on keywords in its filename. You **do not** need to edit the code when adding new experiments; just name your files correctly.

#### 1. File Location
Place all your experimental files (`.csv` or `.xlsx`) inside the `datasets/` folder in the project root.

#### 2. Naming Convention (Automatic Classification)
The script looks for specific keywords in the filename to determine the experiment class.

| Class | Keyword Required in Filename | Example Filename | Description |
| :--- | :--- | :--- | :--- |
| **Replicates** | `replicates` | `assay_replicates_01.csv` | Assays with biological variation (multiple columns). |
| **First Order** | `1storder` | `growth_1storder_A.xlsx` | Smooth, simple decay or growth curves. |
| **Unfinished** | `unfinished` | `data_unfinished_experiment.csv` | Assays where the plateau was not fully reached. |

> **Note:** The matching is case-insensitive (e.g., `REPLICATES` works the same as `replicates`). Files that do not contain any of these keywords will be skipped with a warning.

#### 3. File Content Structure
The content within each file must follow the standard app format:
* **Columns:** Organized in pairs (Time, Response).
* **Replicates:** Placed side-by-side in the same file. The system automatically detects them (e.g., 6 columns = 3 replicates).

**Example (`assay_replicates_01.csv`):**
| Time (Rep 1) | OD (Rep 1) | Time (Rep 2) | OD (Rep 2) |
| :--- | :--- | :--- | :--- |
| 0.0 | 0.05 | 0.0 | 0.04 |
| 1.0 | 0.12 | 1.0 | 0.11 |
| ... | ... | ... | ... |
