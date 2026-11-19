# How to run Kimi K3.1 Pro with inferenceSE (profile "myKimi3")

This guide takes you from a freshly installed Ubuntu computer to a chat window
in your web browser where you talk to the Kimi K3.1 Pro language model, running
entirely on your own machine. No previous Linux knowledge is needed. Every
command is given in full; you copy it, paste it, press Enter.

It is written for this computer, the system the `myKimi3` profile was made for:

| Part | Model |
|---|---|
| Graphics card (GPU) | NVIDIA RTX PRO 6000 Blackwell, 96 GB |
| Processor (CPU) | AMD Ryzen 9 9950X3D |
| Memory (RAM) | 128 GB DDR5-6000 (2 × 64 GB) |
| SSD | WD_BLACK SN8100, 4 TB |
| Mainboard | ASRock X870E Taichi |

---

## Before you start: read this

**What to expect.** The model is 1.56 terabytes, far more than your 96 GB of
graphics memory plus 128 GB of RAM. inferenceSE runs it anyway by reading the
missing parts from the SSD while it works. That makes it slow:

- Starting the program: a few minutes.
- Before the first word of each new answer: a few minutes (the model has to
  read your message).
- Then roughly **one word every 3 to 5 seconds**.

A short answer takes several minutes, a long one an hour or more. This is the
price of running a model of this size at home.

**This setup is new.** The engine has been tested on small models and on a
data-free copy of the Kimi file layout, but **it has never yet been run with
the real Kimi weights or on an NVIDIA graphics card**. You may be the first.
If the answers come out as nonsense, or the program stops with an error, that
is a problem in inferenceSE, not something you did wrong. The section
[If something goes wrong](#if-something-goes-wrong) says what to try and what
to report.

**What you need besides the computer:**

- An internet connection. The model download is 1.56 TB. The table in
  [Step 8](#step-8-download-the-model) shows how long that takes at your
  speed; on many home connections it is a day or more.
- At least **1.7 TB of free space** on the SSD.
- About an hour for Steps 1 to 7, plus the download time.

---

## Contents

0. [Terminal basics](#step-0-terminal-basics)
1. [Firmware (BIOS) settings](#step-1-firmware-bios-settings)
2. [Update Ubuntu](#step-2-update-ubuntu)
3. [Install the NVIDIA graphics driver](#step-3-install-the-nvidia-graphics-driver)
4. [Install the basic tools](#step-4-install-the-basic-tools)
5. [Check the free disk space](#step-5-check-the-free-disk-space)
6. [Download inferenceSE](#step-6-download-inferencese)
7. [Install inferenceSE and its helpers](#step-7-install-inferencese-and-its-helpers)
8. [Download the model](#step-8-download-the-model)
9. [Check the model files](#step-9-check-the-model-files)
10. [Start the chat server](#step-10-start-the-chat-server)
11. [Chat in your browser](#step-11-chat-in-your-browser)
12. [Stop, and start again later](#step-12-stop-and-start-again-later)

Then: [Settings you can change](#settings-you-can-change),
[If something goes wrong](#if-something-goes-wrong),
[What gets installed, and which versions](#what-gets-installed-and-which-versions),
[Removing everything](#removing-everything).

---

## Step 0: Terminal basics

Almost everything is done in the **Terminal**, a window where you type
commands.

- **Open it:** press `Ctrl` + `Alt` + `T`. Or click "Show Apps" (the grid of
  dots in the bottom-left corner) and search for "Terminal".
- **Paste a command:** copy it from this guide, click into the Terminal, press
  `Ctrl` + `Shift` + `V` (note the extra `Shift`; plain `Ctrl` + `V` does not
  work in the Terminal), then press `Enter`.
- **Copy text from the Terminal:** select it with the mouse, press `Ctrl` +
  `Shift` + `C`.
- **Wait for the prompt.** A command is finished when the line ending in `$`
  appears again, for example `anna@kimi-pc:~$`.
- **Stop a running command:** `Ctrl` + `C`.
- **Password:** commands starting with `sudo` change the system and ask for
  *your* login password. While you type it, nothing appears on screen — not
  even dots. That is normal. Type it and press `Enter`.
- **`~` means your home folder**, `/home/<your user name>`. The folder
  `~/models` is the folder `models` inside your home folder.
- **Typing `y` and Enter** answers "yes" when a command asks
  `Do you want to continue? [Y/n]`.

Each command in this guide sits in its own grey box. Run them one at a time,
in order, and wait for each to finish.

---

## Step 1: Firmware (BIOS) settings

These are optional but give you full speed. You change them in the
mainboard's setup screen, which opens when you press `F2` or `Del` right after
switching the computer on.

On the ASRock X870E Taichi:

| Setting | Where (approximately) | Set to | Why |
|---|---|---|---|
| EXPO | OC Tweaker → DRAM Configuration → "EXPO" | **Enabled** (profile 1) | Memory runs at its rated 6000 MT/s instead of a slower default. |
| Above 4G Decoding | Advanced → PCI Configuration | **Enabled** | Lets the system address the GPU's 96 GB properly. |
| Re-Size BAR Support | Advanced → PCI Configuration | **Enabled** | Faster transfers between RAM and the GPU. |

Save with `F10` and let the computer restart. If it fails to start after
enabling EXPO, enter the setup again and disable EXPO; everything still works,
a little slower.

Leave **Secure Boot** as it is. Step 3 explains the one extra screen it may
cause.

---

## Step 2: Update Ubuntu

This guide assumes **Ubuntu 24.04 LTS or 26.04 LTS, Desktop, 64-bit**,
installed normally. (LTS means "long-term support"; these are the versions to
use.) To see which one you have:

```bash
lsb_release -d
```

It prints something like `Description: Ubuntu 24.04.3 LTS`.

Bring everything up to date. This can take a few minutes:

```bash
sudo apt update
```

```bash
sudo apt full-upgrade -y
```

Then restart the computer:

```bash
sudo reboot
```

---

## Step 3: Install the NVIDIA graphics driver

The program needs NVIDIA's own driver to use the graphics card. Your card
belongs to the "Blackwell" generation, which requires **driver version 575 or
newer, in its "open" variant**. Version **580 or newer** is recommended.

Open the Terminal again (`Ctrl` + `Alt` + `T`) and list the drivers Ubuntu
offers for your card:

```bash
sudo ubuntu-drivers list
```

You see lines like `nvidia-driver-580-open`. Install the one Ubuntu
recommends:

```bash
sudo ubuntu-drivers install
```

> **If you see a blue or purple screen about "Secure Boot" or a "MOK key"
> during installation or after the restart:** this only happens on some
> systems. During installation you are asked to choose a password (any, you
> need it once). After the restart a blue screen titled "Perform MOK
> management" appears: choose **Enroll MOK** → **Continue** → **Yes**, type that
> password, choose **Reboot**.

Restart:

```bash
sudo reboot
```

Check that the driver works:

```bash
nvidia-smi
```

You should see a table naming **NVIDIA RTX PRO 6000 Blackwell**, about
**97 GB** of memory (`97887MiB` or similar) and, at the top, the
**Driver Version**.

- If the driver version is **below 575**, install a newer one by name, then
  restart:

  ```bash
  sudo apt install -y nvidia-driver-580-open
  ```

  ```bash
  sudo reboot
  ```

- If you get `command not found` or "couldn't communicate with the NVIDIA
  driver", the driver is not active. Restart once more. If it persists, see
  [If something goes wrong](#if-something-goes-wrong).

You do **not** need to install the "CUDA Toolkit" from NVIDIA's website. The
parts of CUDA the program needs come with PyTorch in Step 7.

---

## Step 4: Install the basic tools

`git` downloads the program, `python3-venv` gives it a private place for its
helper packages, `curl` and `htop` are small utilities.

```bash
sudo apt install -y git python3 python3-venv python3-pip curl htop
```

Check the Python version:

```bash
python3 --version
```

Versions **3.10 to 3.13** are known to work. Ubuntu 24.04 has 3.12. If you
see 3.14 or newer and Step 7 later fails to install PyTorch, follow
[the note on Python versions](#pytorch-fails-to-install-or-your-python-is-too-new).

---

## Step 5: Check the free disk space

The model must go on the fast WD_BLACK SN8100 SSD. Check the free space in
your home folder:

```bash
df -h ~
```

Look at the column **Avail**. You need at least **1.7T** (1.7 terabytes).

On most machines Ubuntu itself is installed on the SN8100, the home folder is
on it too, and this is all you need. **The rest of this guide assumes that.**

> **If Ubuntu is on another drive and the SN8100 is a second, empty drive:**
> you first have to prepare it.
>
> 1. **Make sure there is nothing on it you need. The next steps erase it.**
> 2. Open the app **Disks** (Show Apps → search "Disks").
> 3. Select the 4 TB WD_BLACK drive on the left.
> 4. Click the "⋮" menu at the top right → **Format Disk…** → Erase: "Don't
>    overwrite", Partitioning: "GPT" → **Format**.
> 5. Click **+** under the empty bar → use the full size → Next → Volume Name
>    `data`, Type **Ext4** → **Create**.
> 6. With the new partition selected, click the gear icon → **Edit Mount
>    Options…** → switch off "User Session Defaults" → tick "Mount at system
>    startup" → Mount Point `/data` → **OK**. Then click the ▶ button to mount
>    it.
> 7. In the Terminal, give yourself ownership:
>
>    ```bash
>    sudo chown "$USER": /data
>    ```
>
> From then on, wherever this guide writes `~/models`, use `/data/models`
> instead.

Do **not** put the model on a USB disk or a network drive. They are far too
slow.

---

## Step 6: Download inferenceSE

> **Important.** On GitHub, the program is not on the repository's default
> page ("main"), which holds only a `LICENSE` file. It is on a separate branch
> called `claude/focused-mccarthy-3u3td5`. If you download the default page,
> Step 7 fails with
> `does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found`.
> The command below picks the right branch, and the check at the end of this
> step catches the mistake before it causes trouble.

This copies the program into the folder `~/inferenceSE`:

```bash
cd ~
```

```bash
git clone -b claude/focused-mccarthy-3u3td5 https://github.com/MT-Blachetta/inferenceSE.git
```

(`-b claude/focused-mccarthy-3u3td5` picks the branch. Type it exactly,
including the `-b`. Once the work has been merged into "main", the command
without `-b …` will do.)

If git answers `fatal: destination path 'inferenceSE' already exists`, you
have downloaded it before. Do not skip ahead: use the repair commands at the
end of this step.

> **If git asks for a user name and password**, the repository is private.
> Ask its owner for access, or log in to GitHub in your browser, open the
> repository page, click the branch button that says **main** (top left,
> above the file list), choose `claude/focused-mccarthy-3u3td5`, then click
> **Code → Download ZIP**. Unpack the ZIP into your home folder and rename the
> unpacked folder to `inferenceSE`.

Go into the folder:

```bash
cd ~/inferenceSE
```

**Check that you have the complete program:**

```bash
ls
```

You must see, among others, **`pyproject.toml`**, `inferencese`, `profiles`,
`how-to-run.md` and `README.md`. If you see **only `LICENSE`**, you have the
default page instead of the branch. Repair it, without downloading anything
twice, with these two commands:

```bash
git fetch origin claude/focused-mccarthy-3u3td5
```

```bash
git checkout claude/focused-mccarthy-3u3td5
```

Run `ls` again; `pyproject.toml` must now be there.

If `git fetch` says `not a git repository`, the folder came from a ZIP of the
default page. Delete it and download again:

```bash
cd ~
```

```bash
rm -rf ~/inferenceSE
```

```bash
git clone -b claude/focused-mccarthy-3u3td5 https://github.com/MT-Blachetta/inferenceSE.git
```

```bash
cd ~/inferenceSE
```

(Deleting the folder also deletes the Python environment from Step 7 if you
had already created it; simply do Step 7 again.)

---

## Step 7: Install inferenceSE and its helpers

First create a private Python environment inside the program folder. It keeps
everything the program installs separate from the rest of the system:

```bash
python3 -m venv ~/inferenceSE/.venv
```

**Activate** it:

```bash
source ~/inferenceSE/.venv/bin/activate
```

Your prompt now starts with `(.venv)`. **Every time you open a new Terminal
to work with inferenceSE, run this `source …` command first.**

Update the installer:

```bash
pip install --upgrade pip
```

Install **PyTorch with CUDA support** — the part that does the arithmetic on
the graphics card. It is about 3 GB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install **inferenceSE** itself, with the tokenizer packages it needs for Kimi.
This command must be run **inside the folder `~/inferenceSE`** (the `.` in it
means "the current folder"), so go there first:

```bash
cd ~/inferenceSE
```

```bash
pip install -e ".[tokenizers]"
```

If this ends with `neither 'setup.py' nor 'pyproject.toml' found`, you are
either in the wrong folder or downloaded the wrong branch: go back to the
check at the end of [Step 6](#step-6-download-inferencese).

Install the **Hugging Face download tool**, used in Step 8:

```bash
pip install -U "huggingface_hub[cli,hf_xet]"
```

Now check that Python can see the graphics card:

```bash
python -c "import torch; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU found:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

Expected: `GPU found: True | NVIDIA RTX PRO 6000 Blackwell …` with a CUDA
version of **12.8 or higher**. If you see `False` or an error, stop here and
see [If something goes wrong](#if-something-goes-wrong).

Check that inferenceSE is installed and knows the profile:

```bash
inferencese profiles
```

The list must contain `myKimi3`.

---

## Step 8: Download the model

### Where the files come from

The model's page is <https://huggingface.co/moonshotai/Kimi-K3.1-Pro>. Its
**Files and versions** tab lists everything: 96 large files named
`model-00001-of-000096.safetensors` to `model-00096-of-000096.safetensors`
(most of them 17 GB), plus small files such as `config.json`,
`tiktoken.model` and `tokenizer_config.json`.

You could click each file's download arrow on that tab, but with 96 files of
up to 17 GB that is impractical and a browser cannot resume a broken
download. Use the download tool instead; it fetches exactly the files on
that tab, several at a time, and resumes where it left off.

### Where the files go

Put them in this folder:

```
~/models/Kimi-K3.1-Pro
```

(that is `/home/<your user name>/models/Kimi-K3.1-Pro`; or
`/data/models/Kimi-K3.1-Pro` if you prepared a separate drive in Step 5).
All files go **directly into that one folder**, side by side, not into
subfolders. This folder path is what you hand to the program later.

### Download

Make sure the environment is active (prompt starts with `(.venv)`; if not,
run `source ~/inferenceSE/.venv/bin/activate`). Create the folder:

```bash
mkdir -p ~/models/Kimi-K3.1-Pro
```

Optional but recommended: a free Hugging Face account gives you faster,
less throttled downloads. Create one at <https://huggingface.co/join>, then
open <https://huggingface.co/settings/tokens>, click **Create new token**,
choose type **Read**, create it and copy it. Then:

```bash
hf auth login
```

Paste the token when asked (`Ctrl` + `Shift` + `V`; it stays invisible) and
press Enter. Answer `n` to the question about git credentials.

Start the download:

```bash
hf download moonshotai/Kimi-K3.1-Pro --local-dir ~/models/Kimi-K3.1-Pro
```

This shows progress bars. How long it takes:

| Internet speed | Approximate download time for 1.56 TB |
|---|---|
| 100 Mbit/s | 35 hours |
| 250 Mbit/s | 14 hours |
| 500 Mbit/s | 7 hours |
| 1 Gbit/s | 3.5 hours |

**The computer must stay on and awake.** Switch off automatic suspend in
Settings → Power → "Automatic Suspend" → Off.

**If the download stops** (power cut, Wi-Fi drop, you pressed `Ctrl` + `C`),
run exactly the same `hf download …` command again. Files that are complete
are skipped; the rest continue.

When it is finished, the folder contains, among others:

```
~/models/Kimi-K3.1-Pro/
├── config.json
├── tiktoken.model
├── tokenizer_config.json
├── model-00001-of-000096.safetensors
├── …
├── model-00096-of-000096.safetensors
└── (several other small files, an "assets" folder and a hidden ".cache" folder — leave them)
```

---

## Step 9: Check the model files

Before starting, let inferenceSE read the file headers and count the
tensors. This takes a few seconds and reads no weights:

```bash
inferencese inspect ~/models/Kimi-K3.1-Pro
```

The first line must say exactly:

```
497,340 tensors, 1,454.00 GiB in 96 file(s)
```

(1,454 GiB is the same as 1.56 TB; the two units differ.) If the numbers are
smaller, files are missing: run the `hf download …` command from Step 8 again.

Optionally, see how the program will divide the model between graphics
memory, RAM and SSD:

```bash
inferencese plan myKimi3 --model ~/models/Kimi-K3.1-Pro
```

---

## Step 10: Start the chat server

The chat server is the program that loads the model and offers the chat
window. It runs in a Terminal window and must stay running while you chat.

1. Open a Terminal (`Ctrl` + `Alt` + `T`).
2. Activate the environment:

   ```bash
   source ~/inferenceSE/.venv/bin/activate
   ```

3. Start the server:

   ```bash
   inferencese serve myKimi3 --model ~/models/Kimi-K3.1-Pro --set tokenizer.thinking=false
   ```

What the parts mean:

| Part | Meaning |
|---|---|
| `inferencese` | the program |
| `serve` | start the chat server (other commands: `run`, `plan`, `inspect`, …) |
| `myKimi3` | the optimization profile for Kimi K3.1 Pro on this computer |
| `--model ~/models/Kimi-K3.1-Pro` | **where the model files are** — the folder from Step 8 |
| `--set tokenizer.thinking=false` | the model answers directly instead of first "thinking" at length. Recommended for the first test, because thinking can take thousands of extra words, i.e. hours. Leave this part out to get the model's full reasoning. |

The Terminal shows what happens. Within a few minutes you should see lines
like:

```
[inferenceSE] profile=myKimi3 backend=torch/cuda tensors=497340 model=1454.00 GiB vram-cache=91.00 GiB ram-cache=116.00 GiB policy=plan
[inferenceSE] warm-up: class=lm_head loaded 5/5 blocks
[inferenceSE] warm-up: class=dense loaded 2578/2578 blocks
[inferenceSE] warm-up done in …s …
[inferenceSE] serving myKimi3 on http://127.0.0.1:8080 (OpenAI-compatible API under /v1, chat UI at /)
```

(The exact cache sizes depend on what is free at that moment.) The last line
means it is ready. **Leave this window open.** Closing it, or pressing
`Ctrl` + `C` in it, stops the server.

Check that `backend=torch/cuda` appears in the first line. If it says
`numpy` or `torch` without `/cuda`, the graphics card is not being used: see
[If something goes wrong](#if-something-goes-wrong).

---

## Step 11: Chat in your browser

Open the web browser (Firefox is installed with Ubuntu) and go to:

**<http://127.0.0.1:8080>**

(Type it into the address bar exactly like that and press Enter. It only
works on this computer, and only while the server from Step 10 is running.)

You see the chat window:

- **Message box** at the bottom: type your question. `Enter` sends it,
  `Shift` + `Enter` starts a new line.
- **max tokens**: the longest answer, in tokens (a token is roughly
  three-quarters of a word). The default of 256 is short. Set it to
  **800** for normal answers; more if you want long texts and have the time.
- **temperature**: leave empty for the model's default (1.0). `0` makes the
  answers more predictable and repeatable.
- **system**: an optional standing instruction, for example
  `Answer in German.` or `Be brief.`
- **Send / Stop**: while the model is writing, the button reads **Stop** and
  ends the answer early.
- **Memory** (top right): shows how much of the model is in graphics memory
  and RAM, how much has been read from the SSD, and the speed so far.
- **New chat** (top right): forget the conversation and start over.
- If thinking is switched on, the model's reasoning appears in a grey
  "Reasoning" box above the answer and folds away when the answer starts.

**What happens after you press Send:**

1. The status at the bottom says "reading prompt…" for **a few minutes** while
   the model processes your message. Nothing appears yet. This is normal.
2. Then the answer appears, **a word every few seconds**.
3. Follow-up questions in the same chat are faster to start, because the
   earlier conversation does not have to be read again.

Try a short first question such as *"What is the capital of France? Answer
in one sentence."* If the reply is sensible, everything works.

---

## Step 12: Stop, and start again later

**To stop:** click into the Terminal window of the server and press
`Ctrl` + `C`. The program saves which parts of the model were used most, so
the next start is quicker to get going.

**To start again later** (for example after a restart of the computer), you
only need Steps 10 and 11:

```bash
source ~/inferenceSE/.venv/bin/activate
```

```bash
inferencese serve myKimi3 --model ~/models/Kimi-K3.1-Pro --set tokenizer.thinking=false
```

Then open <http://127.0.0.1:8080>.

### Optional: a start command you can double-click

Create a small script once:

```bash
cat > ~/start-kimi.sh << 'EOF'
#!/bin/bash
source ~/inferenceSE/.venv/bin/activate
( sleep 240 && xdg-open http://127.0.0.1:8080 ) &
inferencese serve myKimi3 --model ~/models/Kimi-K3.1-Pro --set tokenizer.thinking=false
read -p "The server has stopped. Press Enter to close this window."
EOF
```

```bash
chmod +x ~/start-kimi.sh
```

From now on, open a Terminal and type `~/start-kimi.sh`. It starts the
server and opens the chat page in the browser after four minutes. (If the page
says it cannot connect, the server is still loading: wait and press reload.)

### Optional: keep the folder path in the profile

If you do not want to type `--model …` every time, open
`~/inferenceSE/profiles/myKimi3/profile.yaml` in the Text Editor, find the
line that starts with

```
  path: null
```

and replace the word `null` with the full path, for example
`  path: /home/anna/models/Kimi-K3.1-Pro` (your user name instead of `anna`;
keep the two spaces at the beginning). Save. Then
`inferencese serve myKimi3` is enough.

---

## Settings you can change

Add any of these to the `inferencese serve …` command, separated by spaces.

| Add this | Effect |
|---|---|
| `--set tokenizer.thinking=false` | Answer directly, no reasoning phase (fastest). |
| `--set tokenizer.thinking_effort=low` | With thinking on: think briefly. Values `low`, `high`, `max` (default `max`). Use instead of `thinking=false`. |
| `--port 8081` | Use another port if 8080 is taken; the chat page is then <http://127.0.0.1:8081>. |
| `--host 0.0.0.0 --api-key MYSECRET` | Make the chat reachable from other devices in your home network at `http://<this computer's IP>:8080` (find the IP with `hostname -I`). The chat page asks for the key once per browser. Only do this on a network you trust: the connection is not encrypted. |
| `--ram 100GiB` | Use less RAM for the model (default: everything free except 12 GiB). Use this if you want to keep working on the computer while it runs. |
| `--vram 85GiB` | Use less graphics memory (default: everything free except 5 GiB). |
| `--set runtime.pin_host_memory=false` | Turn off a speed optimisation, if the program fails with memory errors at start. |
| `-v` | Log every web request in the Terminal. |

To see all options:

```bash
inferencese serve --help
```

The full list of profile settings is in `profiles/myKimi3/README.md` and
`docs/PROFILES.md`.

### Using it from other programs

The server speaks the same language (API) as OpenAI's service, so programs
that can talk to "an OpenAI-compatible server" can use it. Give them:

- Base URL: `http://127.0.0.1:8080/v1`
- Model: `moonshotai/Kimi-K3.1-Pro`
- API key: anything (or the one you set with `--api-key`)

---

## If something goes wrong

**Look at the Terminal where the server runs first.** Error messages appear
there. Copy them (`Ctrl` + `Shift` + `C`) if you need to ask for help.

#### `nvidia-smi` says "command not found" or cannot talk to the driver

- Restart the computer once more.
- If Secure Boot is on and you skipped the "Enroll MOK" screen, run
  `sudo apt install --reinstall nvidia-driver-580-open`, restart, and this
  time complete the blue screen (Step 3).
- Make sure the monitor is plugged into the **graphics card**, not into the
  mainboard's video outputs.

#### PyTorch says `GPU found: False`

- Check `nvidia-smi` works (above).
- Make sure the environment is active (`(.venv)` at the start of the prompt).
- Reinstall PyTorch with CUDA:

  ```bash
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
  ```

#### PyTorch fails to install, or your Python is too new

If `pip install torch …` says "No matching distribution found", your
Python version is newer than the available PyTorch packages. Use the tool
`uv` to get Python 3.12 just for this program:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close the Terminal, open a new one, then:

```bash
rm -rf ~/inferenceSE/.venv
```

```bash
uv venv --python 3.12 --seed ~/inferenceSE/.venv
```

Then continue Step 7 from `source ~/inferenceSE/.venv/bin/activate`.

#### `does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found`

This appears at `pip install -e ".[tokenizers]"` (Step 7). Two possible
causes:

1. **The wrong folder.** The command must run in `~/inferenceSE`. Run
   `cd ~/inferenceSE` and try again.
2. **The wrong branch.** Run `ls ~/inferenceSE`. If you see only `LICENSE`,
   you downloaded the repository's default page, not the branch with the
   program. Repair it:

   ```bash
   cd ~/inferenceSE
   ```

   ```bash
   git fetch origin claude/focused-mccarthy-3u3td5
   ```

   ```bash
   git checkout claude/focused-mccarthy-3u3td5
   ```

   Then run the `pip install -e ".[tokenizers]"` command again. The Python
   environment you already created stays valid. If `git fetch` says
   `not a git repository`, follow the delete-and-download instructions at the
   end of [Step 6](#step-6-download-inferencese).

#### `inferencese: command not found`

The environment is not active. Run
`source ~/inferenceSE/.venv/bin/activate`.

#### "model path … does not exist" or "no model files … found"

The path after `--model` is wrong. Check it:

```bash
ls ~/models/Kimi-K3.1-Pro
```

You must see the `model-000…safetensors` files listed directly. If they are
in a subfolder, give that subfolder as the path.

#### The first line says `backend=numpy` or no `/cuda`

The graphics card is not used. Fix PyTorch (above). You can force the
graphics card with `--backend torch --device cuda`; the error message then
tells you why it does not work.

#### "Address already in use"

Another program, or a server you started earlier, uses port 8080. Close the
other server window, or add `--port 8081` and use <http://127.0.0.1:8081>.

#### The program stops with "out of memory" (CUDA or RAM)

Leave more room: add `--vram 80GiB --ram 96GiB`. Close other programs,
especially browsers with many tabs. If it happens right at start, also add
`--set runtime.pin_host_memory=false`.

#### The page shows nothing for a long time after Send

Expected for the first few minutes ("reading prompt…"). Watch the SSD
activity: open a second Terminal and run `htop`; the `python` process should
be busy. If nothing appears after 15 minutes, look at the server Terminal for
errors.

#### The answers are nonsense

Use `temperature` `0` in the chat window and ask a simple question. If the
reply is still meaningless text, the engine computes the model incorrectly.
This is the first thing that needs checking on real hardware (see "Before
you start"). Please report it with the output of these two commands:

```bash
inferencese hardware
```

```bash
inferencese run myKimi3 --model ~/models/Kimi-K3.1-Pro --chat -p "What is the capital of France?" -n 30 --temperature 0 --set tokenizer.thinking=false --stats
```

#### Very slow, much slower than a word every 5 seconds

- Check that the model is on the SN8100 (`df -h ~/models/Kimi-K3.1-Pro`
  shows `/dev/nvme…`), not on a USB disk.
- Check that the graphics card is used (`backend=torch/cuda`, and
  `nvidia-smi` shows a `python` process using about 90 GB).
- Close other programs so more RAM is free.

---

## What gets installed, and which versions

| Component | Version | Installed in | Why |
|---|---|---|---|
| Ubuntu Desktop | 24.04 LTS or 26.04 LTS, 64-bit | — | the operating system |
| NVIDIA driver | 580 or newer, **open** variant (minimum 575) | Step 3 | required for Blackwell graphics cards |
| git, python3, python3-venv, python3-pip, curl, htop | Ubuntu's versions | Step 4 | download and run the program |
| Python | 3.10 – 3.13 (Ubuntu 24.04: 3.12) | comes with Ubuntu | the language inferenceSE is written in |
| PyTorch | 2.7 or newer, built for CUDA 12.8 or newer ("cu128") | Step 7 | arithmetic on the graphics card; includes the CUDA libraries |
| inferenceSE | this repository | Step 7 | the inference engine and chat server |
| numpy, PyYAML | current | Step 7, automatically | arrays, configuration files |
| tiktoken, tokenizers, jinja2 | current | Step 7 (`[tokenizers]`) | turn text into the model's tokens and back |
| huggingface_hub (`hf` command) with hf_xet | current | Step 7 | download the model |

Not needed: the NVIDIA CUDA Toolkit, cuDNN, Docker, a C compiler, or
vLLM / llama.cpp.

Disk space used: model 1.56 TB, Python environment about 8 GB, and a working
file for long conversations (`~/.cache/inferencese/myKimi3/swap.bin`) that
grows only when a conversation no longer fits in memory.

---

## Removing everything

Stop the server first. Then:

```bash
rm -rf ~/models/Kimi-K3.1-Pro
```

```bash
rm -rf ~/inferenceSE ~/.cache/inferencese ~/start-kimi.sh
```

The NVIDIA driver can stay; it is useful for the graphics card anyway.
