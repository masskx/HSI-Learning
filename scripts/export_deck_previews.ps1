# Export PPTX deck slides to PNG via desktop PowerPoint COM automation.
# Windows-only; requires an installed desktop PowerPoint (not verified with WPS).
#
#   powershell -File scripts\export_deck_previews.ps1 -Only L01,L09
#
# Exports go to slides\previews\com-export\<deck>-<stamp>\NN.png at 1600x900.
# Rasterized pages still require human visual inspection; this script only
# proves that PowerPoint can open and render the decks.
param(
    [string]$DecksDir = "slides\decks",
    [string]$OutDir = "slides\previews\com-export",
    [string[]]$Only = @()
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$decks = Get-ChildItem -Path (Join-Path $repo $DecksDir) -Filter 'L*.pptx' | Sort-Object Name
if ($Only) { $decks = $decks | Where-Object { $Only -contains ($_.Name.Split('-')[0]) } }
if (-not $decks) { throw 'No matching PPTX decks found.' }
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$app = New-Object -ComObject PowerPoint.Application
$exported = 0
try {
    foreach ($deck in $decks) {
        $out = Join-Path $repo "$OutDir\$($deck.BaseName)-$stamp"
        New-Item -ItemType Directory -Force $out | Out-Null
        $pres = $app.Presentations.Open($deck.FullName, $true, $false, $false)
        try {
            $i = 1
            foreach ($s in $pres.Slides) {
                $s.Export((Join-Path $out ('{0:d2}.png' -f $i)), 'PNG', 1600, 900)
                $i++
            }
            Write-Output ("{0}: {1} slides -> {2}" -f $deck.Name, ($i - 1), $out)
            $exported += $i - 1
        }
        finally { $pres.Close() }
    }
}
finally { $app.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) }
Write-Output "Exported $exported slides total."
