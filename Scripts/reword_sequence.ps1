param(
    [string]$todoPath
)

(Get-Content -Path $todoPath) `
    -replace 'pick 55c0e0d', 'reword 55c0e0d' `
    -replace 'pick 696ad60', 'reword 696ad60' `
    -replace 'pick 280f21b', 'reword 280f21b' `
    -replace 'pick 8a934b1', 'reword 8a934b1' |
    Set-Content -Path $todoPath

