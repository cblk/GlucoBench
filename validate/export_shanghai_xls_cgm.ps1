param()

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$archivePath = Join-Path $repoRoot 'output\external_datasets\raw\shanghai_t2dm\diabetes_datasets.zip'
$derivedRoot = Join-Path $repoRoot 'output\external_datasets\derived\shanghai_xls_ace'
$workingXls = Join-Path $derivedRoot 'current.xls'
$outputCsv = Join-Path $repoRoot 'output\external_shanghai_xls_cgm.csv'
$summaryJson = Join-Path $repoRoot 'output\external_shanghai_xls_export_summary.json'

New-Item -ItemType Directory -Path $derivedRoot -Force | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem

$archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
$entries = @($archive.Entries | Where-Object {
    $_.FullName.StartsWith('Shanghai_T2DM/', [StringComparison]::Ordinal) -and
    $_.FullName.EndsWith('.xls', [StringComparison]::OrdinalIgnoreCase)
})

$utf8 = New-Object System.Text.UTF8Encoding($false)
$writer = New-Object System.IO.StreamWriter($outputCsv, $false, $utf8)
$writer.WriteLine('record_id,timestamp,glucose_mgdl')

$successful = 0
$rowCount = 0
$failures = @()

try {
    foreach ($entry in $entries) {
        $recordId = [IO.Path]::GetFileNameWithoutExtension($entry.Name)
        try {
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $workingXls, $true)
            $connection = New-Object -ComObject ADODB.Connection
            try {
                $connection.Open("Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$workingXls;Extended Properties='Excel 8.0;HDR=YES;IMEX=1'")
                $schema = $connection.OpenSchema(20)
                try {
                    $sheetName = $null
                    while (-not $schema.EOF) {
                        $candidate = [string]$schema.Fields.Item('TABLE_NAME').Value
                        $normalized = $candidate.Trim("'")
                        if ($normalized.EndsWith('$') -and -not $normalized.Contains('FilterDatabase')) {
                            $sheetName = $normalized
                            break
                        }
                        $schema.MoveNext()
                    }
                }
                finally {
                    $schema.Close()
                }
                if (-not $sheetName) { throw 'No worksheet table found' }

                $command = New-Object -ComObject ADODB.Command
                $command.ActiveConnection = $connection
                $command.CommandText = "SELECT * FROM [$sheetName]"
                $records = $command.Execute()
                try {
                    $dateIndex = $null
                    $cgmIndex = $null
                    for ($fieldIndex = 0; $fieldIndex -lt $records.Fields.Count; $fieldIndex += 1) {
                        $fieldName = ([string]$records.Fields.Item($fieldIndex).Name).Trim()
                        if ($fieldName.Equals('Date', [StringComparison]::OrdinalIgnoreCase)) { $dateIndex = $fieldIndex }
                        if ($fieldName.StartsWith('CGM', [StringComparison]::OrdinalIgnoreCase)) { $cgmIndex = $fieldIndex }
                    }
                    if ($null -eq $dateIndex -or $null -eq $cgmIndex) { throw 'Date or CGM field not found' }
                    while (-not $records.EOF) {
                        $stamp = $records.Fields.Item($dateIndex).Value
                        $glucose = $records.Fields.Item($cgmIndex).Value
                        if ($null -ne $stamp -and $null -ne $glucose -and -not [Convert]::IsDBNull($stamp) -and -not [Convert]::IsDBNull($glucose)) {
                            $parsedStamp = [DateTime]$stamp
                            $parsedGlucose = [double]$glucose
                            if (-not [double]::IsNaN($parsedGlucose) -and -not [double]::IsInfinity($parsedGlucose)) {
                                $writer.WriteLine(('{0},{1},{2}' -f $recordId, $parsedStamp.ToString('yyyy-MM-ddTHH:mm:ss'), $parsedGlucose.ToString('R', [Globalization.CultureInfo]::InvariantCulture)))
                                $rowCount += 1
                            }
                        }
                        $records.MoveNext()
                    }
                }
                finally {
                    $records.Close()
                }
                $successful += 1
            }
            finally {
                if ($connection.State -ne 0) { $connection.Close() }
                [Runtime.InteropServices.Marshal]::FinalReleaseComObject($connection) | Out-Null
            }
        }
        catch {
            $failures += [pscustomobject]@{ record_id = $recordId; error = $_.Exception.Message }
        }
    }
}
finally {
    $writer.Close()
    $archive.Dispose()
}

$summary = [ordered]@{
    source_archive = $archivePath
    xls_entries = $entries.Count
    successful_workbooks = $successful
    failed_workbooks = $failures.Count
    exported_cgm_rows = $rowCount
    output_csv = $outputCsv
    failures = $failures
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryJson -Encoding utf8
$summary | ConvertTo-Json -Depth 5
