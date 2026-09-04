<#
Install MamboDubb from source on Windows. Ends with the studio serving on
127.0.0.1:4400, which is the same editor the Mac desktop app wraps.

  powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.ps1 | iex"

There is no -setup.exe and no .msi to download, and there cannot be one until the
account-level GitHub Actions block is lifted (docs/CROSS_PLATFORM.md). So the
install is the from-source route, done for you: fetch the source at the latest
release tag, put uv on the machine, resolve the Python side, put the built web UI
in place, and start the server.

Nothing here downloads a model. That is the Setup screen's job on first run, and
it is about 25 GB.

Environment overrides, all optional:
  MAMBODUBB_DIR      where the checkout goes (default: $HOME\MamboDubb)
  MAMBODUBB_REF      tag or branch to install (default: the latest release)
  MAMBODUBB_PORT     port to serve on (default: 4400)
  MAMBODUBB_START=0  install and stop, do not start the server
  MAMBODUBB_UI_TARBALL  a local prebuilt UI archive instead of the release one

Piped into `iex`, so it has no param block and no $PSScriptRoot, and it must
never ask a question: there is no console to answer on.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Invoke-WebRequest is an order of magnitude slower with the progress bar on,
# and a several-hundred-MB download is where that stops being a curiosity.
$ProgressPreference = 'SilentlyContinue'
# PowerShell 7.4 and later turn a non-zero exit code from a native command into a
# terminating error, which with the 'Stop' above would end the install on the one
# case that is not a failure at all: winget answers non-zero for "already
# installed". Every native call below checks $LASTEXITCODE itself where the exit
# code means something, so the automatic version is turned off rather than
# duplicated. On Windows PowerShell 5.1 this variable does not exist and setting
# it is inert, which is why it is set unconditionally instead of probed for.
$PSNativeCommandUseErrorActionPreference = $false

$Repo = 'maxmelichov/MamboDubb'
$UiAsset = 'mambodubb-ui-dist.tar.gz'
# Pinned rather than "latest": this is only the fallback for building the UI when
# the release carries no prebuilt one, and a version that moves under the script
# is a build that breaks on a Tuesday for no reason anyone can see.
$NodeVersion = 'v22.20.0'

function Get-Setting($name, $fallback) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $fallback }
    return $value
}

function Info($message) { Write-Host $message }
function Die($message) { Write-Host "error: $message" -ForegroundColor Red; exit 1 }
function Have($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

# winget puts a tool on PATH for processes started *after* it, so this process
# still cannot see git.exe or ffmpeg.exe it just installed. Rebuilding PATH from
# the registry is what makes an install usable in the same run. WINDOWS.md tells
# a human to open a new terminal for the same reason.
function Update-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
}

# $IsWindows is deliberately not used: it does not exist in Windows PowerShell
# 5.1, and under Set-StrictMode reading a variable that does not exist is an
# error, so the platform check would fail on exactly the shell most Windows
# machines run this in. $env:OS is set on every Windows since NT.
if ($env:OS -ne 'Windows_NT') {
    Die 'this installer is for Windows; on Linux or macOS use install-server.sh'
}

# A 32-bit PowerShell on 64-bit Windows reports x86 and records the real machine
# in PROCESSOR_ARCHITEW6432, so the second variable wins wherever it is set.
$arch = $env:PROCESSOR_ARCHITEW6432
if ([string]::IsNullOrWhiteSpace($arch)) { $arch = $env:PROCESSOR_ARCHITECTURE }
#
# x64 only, and ARM64 is refused here rather than four steps later. astral does
# publish an aarch64 uv and Node does publish an arm64 build, which is why this
# used to accept it, but the sentence it printed ("x64 and arm64 are the ones
# with wheels") was about the wrong wheels. The ones that decide are torch's,
# and uv.lock carries `win_amd64` and nothing else for Windows because
# download.pytorch.org and PyPI both publish no `win_arm64` torch at all. So the
# old path installed git, ffmpeg, sox, uv and Node, spent several minutes on it,
# and then died inside `uv sync` on a resolution error naming a package the user
# never asked for. Refusing in the first ten seconds, and saying which
# dependency is the one that cannot be satisfied, is the whole of the fix: there
# is nothing this script could do differently on that machine.
switch ($arch) {
    'AMD64' { $UvTriple = 'x86_64-pc-windows-msvc'; $NodeArch = 'x64' }
    'ARM64' {
        Die ("Windows on ARM is not supported: PyTorch publishes no win_arm64 " +
             "wheel, so the dependency resolution this installer runs cannot " +
             "succeed on this machine whatever it installs first. An Intel or " +
             "AMD machine, or an ARM Mac, will work.")
    }
    default { Die "unsupported architecture: $arch (x64 is the only one with wheels)" }
}

# Read after the platform guard, not before: the default install directory is
# built from $env:USERPROFILE, which is null anywhere but Windows, and a
# Join-Path exception is a worse way to learn you pasted the wrong command.
$Dir = Get-Setting 'MAMBODUBB_DIR' (Join-Path $env:USERPROFILE 'MamboDubb')
$Port = Get-Setting 'MAMBODUBB_PORT' '4400'
$Ref = Get-Setting 'MAMBODUBB_REF' ''

$Work = Join-Path ([IO.Path]::GetTempPath()) ("mambodubb-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $Work -Force | Out-Null

function Get-File($url, $path) {
    Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
}

function Test-Sha256($path, $expected) {
    $actual = (Get-FileHash -Path $path -Algorithm SHA256).Hash
    return $actual -ieq $expected
}

try {
    # -----------------------------------------------------------------------
    # System tools: git to fetch the source, ffmpeg and sox to run a dub.
    # -----------------------------------------------------------------------
    #
    # The winget ids are the ones dubbing/tools.py names, and they have to stay
    # the same ids: the app's Setup screen runs exactly these lines, and an
    # installer that installed a different ffmpeg would leave Setup reporting a
    # machine that does not exist. tools.py also records that winget, unlike
    # `sudo apt-get`, finishes without asking anything, which is what makes it
    # safe to run from a script nobody is watching.
    $wingetIds = [ordered]@{
        git    = 'Git.Git'
        ffmpeg = 'Gyan.FFmpeg'
        sox    = 'ChrisBagwell.SoX'
    }
    $missing = @($wingetIds.Keys | Where-Object { -not (Have $_) })
    if ($missing.Count -gt 0) {
        if (Have 'winget') {
            foreach ($tool in $missing) {
                Info "Installing $tool"
                winget install --id $wingetIds[$tool] -e --accept-source-agreements --accept-package-agreements
                # winget reports "already installed" as a failure exit code, and
                # a tool that is present is the outcome we wanted either way, so
                # the check below is on the tool and not on this exit code.
            }
            Update-PathFromRegistry
        }
        else {
            Info ''
            Info 'winget is not on this machine, so these have to be installed by hand:'
            foreach ($tool in $missing) {
                Info ("   winget install --id " + $wingetIds[$tool] + " -e")
            }
            Info ''
        }
    }

    # git is the one that cannot wait: pyproject pins third_party/Qwen3-TTS as a
    # path dependency, so a source tree without that submodule can never
    # `uv sync`, and a zip download of the repo has no submodule in it.
    if (-not (Have 'git')) {
        Die 'git is required to fetch the source with its submodule. Install it (winget install --id Git.Git -e), open a new terminal, and rerun.'
    }
    foreach ($tool in @('ffmpeg', 'sox')) {
        if (-not (Have $tool)) {
            Info "warning: $tool is missing. The editor still opens; a dub will fail without it."
        }
    }

    # -----------------------------------------------------------------------
    # The source, at the tag the latest release was cut from.
    # -----------------------------------------------------------------------
    #
    # A tag rather than main, because the prebuilt UI below is a release asset:
    # tying both to the same tag is what keeps the web UI and the API it talks
    # to in step.
    if ([string]::IsNullOrWhiteSpace($Ref)) {
        try {
            $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
            $Ref = $release.tag_name
        }
        catch {
            $Ref = 'main'
        }
    }
    if ([string]::IsNullOrWhiteSpace($Ref)) { $Ref = 'main' }

    if (Test-Path (Join-Path $Dir '.git')) {
        Info "Updating the checkout in $Dir to $Ref"
        git -C $Dir fetch --depth 1 origin $Ref
        if ($LASTEXITCODE -ne 0) { Die "could not fetch $Ref" }
        git -C $Dir checkout --detach FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { Die "could not check out $Ref; move $Dir aside and rerun" }
        git -C $Dir submodule update --init --recursive
        if ($LASTEXITCODE -ne 0) { Die 'submodule checkout failed' }
    }
    elseif (Test-Path $Dir) {
        Die "$Dir exists and is not a git checkout; set MAMBODUBB_DIR to somewhere else"
    }
    else {
        Info "Cloning $Repo ($Ref) into $Dir"
        git clone --depth 1 --branch $Ref --recurse-submodules "https://github.com/$Repo.git" $Dir
        if ($LASTEXITCODE -ne 0) { Die 'clone failed' }
    }

    if (-not (Test-Path (Join-Path $Dir 'third_party\Qwen3-TTS\pyproject.toml'))) {
        Die "third_party\Qwen3-TTS is empty; run: git -C $Dir submodule update --init --recursive"
    }

    Set-Location $Dir

    # -----------------------------------------------------------------------
    # uv, which brings its own Python.
    # -----------------------------------------------------------------------
    $uvHome = Join-Path $env:USERPROFILE '.local\bin'
    $Uv = $null
    if (Have 'uv') { $Uv = (Get-Command uv).Source }
    elseif (Test-Path (Join-Path $uvHome 'uv.exe')) { $Uv = Join-Path $uvHome 'uv.exe' }

    if (-not $Uv) {
        Info "Installing uv ($UvTriple)"
        $base = "https://github.com/astral-sh/uv/releases/latest/download/uv-$UvTriple.zip"
        $zip = Join-Path $Work 'uv.zip'
        Get-File $base $zip
        # The published .sha256 travels from the same host as the archive, so it
        # proves the transfer and not the publisher. That is exactly the check
        # astral's own install script makes, and it is strictly better than
        # piping an unpinned script straight into a shell, which is the
        # alternative on offer.
        try {
            $sums = Join-Path $Work 'uv.sha256'
            Get-File "$base.sha256" $sums
            $expected = ((Get-Content $sums -Raw).Trim() -split '\s+')[0]
            if (-not (Test-Sha256 $zip $expected)) {
                Die 'uv download failed its checksum; refusing to install it'
            }
        }
        catch {
            Info 'warning: could not fetch uv''s published checksum; continuing'
        }
        $unpacked = Join-Path $Work 'uv'
        Expand-Archive -Path $zip -DestinationPath $unpacked -Force
        $exe = Get-ChildItem -Path $unpacked -Filter 'uv.exe' -Recurse | Select-Object -First 1
        if (-not $exe) { Die 'the uv archive did not contain uv.exe' }
        New-Item -ItemType Directory -Path $uvHome -Force | Out-Null
        Copy-Item $exe.FullName (Join-Path $uvHome 'uv.exe') -Force
        $uvx = Get-ChildItem -Path $unpacked -Filter 'uvx.exe' -Recurse | Select-Object -First 1
        if ($uvx) { Copy-Item $uvx.FullName (Join-Path $uvHome 'uvx.exe') -Force }
        $Uv = Join-Path $uvHome 'uv.exe'
    }
    Info "Using uv at $Uv"

    # `--extra app` rather than a plain `uv sync`. fastapi, uvicorn, httpx and
    # python-multipart do land today either way, as transitive dependencies of
    # the gradio that sits under qwen-tts, but that is an accident of somebody
    # else's dependency tree and not a promise. The `app` extra is where this
    # server's own dependencies are declared, and asking for it by name is what
    # keeps the install working on the day gradio drops one of them.
    # `--extra cuda` on top, when there is a card to use it. PyPI's `torch` wheel
    # is a CUDA build on Linux and a CPU-only build on Windows, so without this
    # every model runs on the CPU on a machine that has a GPU: one real user's
    # stem separation took sixteen hours, and nothing in the log said why,
    # because a CPU-only torch is not an error anywhere. The extra also carries
    # the cuBLAS and cuDNN wheels CTranslate2 needs and does not bundle.
    #
    # `nvidia-smi` is the probe for the same reason `dubbing_app/setup.py` uses
    # it: it ships with the driver, it is on PATH on every machine that has one,
    # and asking it costs nothing next to importing torch to find out. A card
    # with no driver is not a card this software can use either way.
    #
    # This script used to print the CUDA swap as advice and leave it to the
    # user. Two things were wrong with that: the version it named no longer
    # exists on download.pytorch.org, and `uv pip install` is undone by the next
    # `uv run`, which re-syncs the venv to the lockfile. The extra is in the
    # lockfile, so it survives.
    $Extras = @('--extra', 'app')
    $HasNvidia = Have 'nvidia-smi'
    # Only when the checkout actually declares the extra: this script is fetched
    # from main but installs the latest release *tag*, and a tag cut before the
    # extra existed (v0.5.0) makes `uv sync --extra cuda` an error, which killed
    # every fresh Windows install on a machine with a card, the machines the
    # extra was invented for. Asking pyproject is what keeps one script correct
    # against every tag it might check out.
    $HasCudaExtra = Select-String -Path (Join-Path $Dir 'pyproject.toml') `
        -Pattern '^cuda\s*=' -Quiet
    $UseCuda = $HasNvidia -and $HasCudaExtra
    if ($UseCuda) {
        Info 'NVIDIA driver found; including the CUDA wheels (a few GB more)'
        $Extras += @('--extra', 'cuda')
    }
    elseif ($HasNvidia) {
        Info "NVIDIA driver found, but the $Ref release predates the CUDA wheel option,"
        Info 'so this install runs the models on the CPU. Rerun this script after the'
        Info 'next release to switch to the CUDA wheels.'
    }
    else {
        Info 'No nvidia-smi on this machine, so the CPU-only wheels are the right ones.'
    }

    Info 'Resolving Python dependencies (several GB the first time)'
    & $Uv sync @Extras
    if ($LASTEXITCODE -ne 0) { Die 'uv sync failed' }

    # The translator is a second uv project with a second torch (translator\
    # pyproject.toml explains why it cannot share the main venv). Its own
    # lockfile sends Windows to the CUDA build of torch, so no flag is needed
    # here; it is synced now rather than left for the first translate stage,
    # which is otherwise where a multi-gigabyte download would land, in the
    # middle of a run.
    Info 'Resolving the translator venv (the 12B translation model runs there)'
    & $Uv sync --project (Join-Path $Dir 'translator')
    if ($LASTEXITCODE -ne 0) { Die 'uv sync for the translator failed' }

    # Say out loud whether it worked. A GPU install that silently did not take
    # is the exact failure this whole block exists to end, so it is not allowed
    # to end in a guess.
    if ($UseCuda) {
        # --no-sync, because a bare `uv run` re-syncs to the lockfile without
        # the extras that were just installed, and removes what they brought:
        # the CUDA torchaudio and the cuBLAS and cuDNN wheels the speech
        # recogniser loads. The check that proves the install took must not be
        # the thing that takes part of it away.
        $cudaOk = & $Uv run --no-sync python -c "import torch; print(torch.cuda.is_available())" 2>$null
        if ("$cudaOk".Trim() -eq 'True') {
            Info 'CUDA is available to torch.'
        }
        else {
            Info 'warning: nvidia-smi is present but torch still reports no CUDA device.'
            Info '  Usually a driver too old for CUDA 12.6. Update the NVIDIA driver and'
            Info "  rerun this script; docs\WINDOWS.md has the details."
        }
    }

    # -----------------------------------------------------------------------
    # The web UI.
    # -----------------------------------------------------------------------
    #
    # The server serves app\ui\dist by default, so getting the built UI into
    # that directory is the whole job. Downloading it is the fast path and the
    # reason Node is not a requirement of this install; building it is the
    # fallback for a release that predates the asset, and that fallback
    # provisions its own Node rather than sending the user away to install one.
    $Dist = Join-Path $Dir 'app\ui\dist'
    # Declared here so the closing message can name where the UI came from even
    # under Set-StrictMode, which treats reading a never-assigned variable as an
    # error rather than as an empty string.
    $UiSource = ''

    function Install-PrebuiltUi {
        $archive = Join-Path $Work 'ui.tar.gz'
        $local = [Environment]::GetEnvironmentVariable('MAMBODUBB_UI_TARBALL')
        if (-not [string]::IsNullOrWhiteSpace($local)) {
            Copy-Item $local $archive -Force
            $script:UiSource = $local
        }
        else {
            Get-File "https://github.com/$Repo/releases/download/$Ref/$UiAsset" $archive
            $script:UiSource = "the $Ref release"
        }
        New-Item -ItemType Directory -Path $Dist -Force | Out-Null
        # tar.exe has shipped in Windows 10 1803 and later, which is the same
        # floor the desktop app's WebView2 requirement already sets.
        tar -xzf $archive -C $Dist
        if ($LASTEXITCODE -ne 0) { throw 'could not unpack the UI archive' }
        if (-not (Test-Path (Join-Path $Dist 'index.html'))) { throw 'the UI archive had no index.html' }
    }

    function Install-Node {
        if ((Have 'node') -and (Have 'pnpm')) { return }
        $name = "node-$NodeVersion-win-$NodeArch"
        $nodeDir = Join-Path $Dir ".tools\$name"
        if (-not (Test-Path (Join-Path $nodeDir 'pnpm.cmd'))) {
            Info "Installing Node $NodeVersion under $Dir\.tools (only to build the UI)"
            $zip = Join-Path $Work 'node.zip'
            Get-File "https://nodejs.org/dist/$NodeVersion/$name.zip" $zip
            $sums = Join-Path $Work 'node.sums'
            Get-File "https://nodejs.org/dist/$NodeVersion/SHASUMS256.txt" $sums
            $line = Get-Content $sums | Where-Object { $_ -match "\s$([regex]::Escape($name)).zip$" } | Select-Object -First 1
            if ($line) {
                $expected = ($line -split '\s+')[0]
                if (-not (Test-Sha256 $zip $expected)) { throw 'Node download failed its checksum' }
            }
            New-Item -ItemType Directory -Path (Join-Path $Dir '.tools') -Force | Out-Null
            if (Test-Path $nodeDir) { Remove-Item $nodeDir -Recurse -Force }
            Expand-Archive -Path $zip -DestinationPath (Join-Path $Dir '.tools') -Force
            # PATH first, then corepack, for the same reason the POSIX installer
            # does it in that order: corepack is a Node script, and it has to be
            # able to find the Node it shipped next to.
            $env:Path = "$nodeDir;$env:Path"
            # corepack ships with every Node 16 and later, and is how pnpm is
            # meant to arrive; no global npm install and no network beyond it.
            corepack enable pnpm --install-directory $nodeDir
            if ($LASTEXITCODE -ne 0) { throw 'corepack could not install pnpm' }
        }
        $env:Path = "$nodeDir;$env:Path"
    }

    function Build-Ui {
        Install-Node
        Info 'Building the web UI from source'
        Push-Location (Join-Path $Dir 'app\ui')
        try {
            pnpm install --frozen-lockfile
            if ($LASTEXITCODE -ne 0) { throw 'pnpm install failed' }
            pnpm build
            if ($LASTEXITCODE -ne 0) { throw 'pnpm build failed' }
        }
        finally { Pop-Location }
    }

    if (Test-Path (Join-Path $Dist 'index.html')) {
        Info 'Web UI already built.'
    }
    else {
        try {
            Install-PrebuiltUi
            Info "Web UI installed from $UiSource."
        }
        catch {
            Info 'No prebuilt web UI on that release; building it from source.'
            try {
                Build-Ui
                Info 'Web UI built from source.'
            }
            catch {
                Die "could not install the web UI ($_); see docs/SERVER.md for the manual build"
            }
        }
    }

    # -----------------------------------------------------------------------
    # Serve.
    # -----------------------------------------------------------------------
    Info ''
    Info "Installed in $Dir"
    Info 'Start it again later with:'
    Info "    cd $Dir; & `"$Uv`" run $($Extras -join ' ') mambodubb --port $Port"
    Info ''
    if ($UseCuda) {
        Info 'Keep `--extra app --extra cuda` on every uv sync and uv run. A bare one'
        Info 'syncs to the lockfile without the extra and removes what it brought: the'
        Info 'CUDA torchaudio and the cuBLAS and cuDNN wheels the speech recogniser loads.'
        Info ''
    }
    Info 'First run: open Setup in the editor and press Install everything. That is'
    Info 'about 25 GB of models, and it resumes where it left off if you interrupt it.'
    Info ''

    if ((Get-Setting 'MAMBODUBB_START' '1') -eq '0') {
        Info 'MAMBODUBB_START=0, so the server was not started.'
        exit 0
    }

    # No Start-Process on the URL: the server takes a moment to bind, and a
    # browser opened first lands on a connection refused page that reads like a
    # failed install. The address is printed instead, once, and it stays true.
    Info "Starting the studio on http://127.0.0.1:$Port (Ctrl-C to stop)"
    & $Uv run @Extras mambodubb --port $Port
}
finally {
    Remove-Item $Work -Recurse -Force -ErrorAction SilentlyContinue
}
