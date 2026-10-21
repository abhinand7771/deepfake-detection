param(
    [string]$OutputRoot = "ffpp",
    [string]$DatasetRoot = "dataset",
    [string]$Dataset = "original",
    [string[]]$Methods = @("Deepfakes","Face2Face","FaceSwap","FaceShifter","NeuralTextures"),
    [int]$NumVideos = 5,
    [int]$FrameStep = 10,
    [int]$MaxFramesPerVideo = 40,
    [int]$MinFaceSize = 80,
    [switch]$Reset
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "[1/2] Downloading authorized FaceForensics++ videos..."
$downloadArgs = @($OutputRoot, "--server", "EU2", "--dataset", $Dataset, "--compression", "c40", "--type", "videos", "--num_videos", $NumVideos.ToString())
python download.py @downloadArgs

Write-Host "[2/2] Organizing frames into dataset/real and dataset/fake..."
$methodList = @()
foreach ($method in $Methods) { $methodList += $method }
$prepareArgs = @("--ffpp-root", $OutputRoot, "--dataset-root", $DatasetRoot, "--frame-step", $FrameStep.ToString(), "--max-frames-per-video", $MaxFramesPerVideo.ToString(), "--min-face-size", $MinFaceSize.ToString())
if ($Reset) { $prepareArgs += "--reset" }
foreach ($method in $methodList) { $prepareArgs += "--methods"; $prepareArgs += $method }
python prepare_ffpp_dataset.py @prepareArgs

Write-Host ""
Write-Host "Dataset prep complete."
Write-Host "Next step: python train.py --dataset-dir dataset --device cpu"
