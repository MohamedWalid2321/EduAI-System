$ErrorActionPreference = 'Stop'

try {
    # Step 1: Login
    $loginBody = '{"email":"student@example.com","password":"P@ssword123"}'
    $loginRes = Invoke-RestMethod -Uri 'https://localhost:7289/api/Authuantication/login' `
        -Method POST `
        -ContentType 'application/json' `
        -Body $loginBody `
        -SkipCertificateCheck

    Write-Host '=== LOGIN SUCCESS ==='
    $token = $loginRes.token
    Write-Host "Token obtained: $($token.Substring(0, [Math]::Min(30, $token.Length)))..."

    # Step 2: Try quiz codes
    $codes = @('EXAM2026', 'TEST', 'QUIZ001', 'demo', 'DEMO')
    foreach ($code in $codes) {
        try {
            $headers = @{ Authorization = "Bearer $token" }
            $attemptRes = Invoke-RestMethod -Uri "https://localhost:7289/api/QuizAttempts/attempt/$code" `
                -Method GET `
                -Headers $headers `
                -SkipCertificateCheck
            Write-Host "=== ATTEMPT ($code) SUCCESS ==="
            $json = $attemptRes | ConvertTo-Json -Depth 10
            Write-Host $json

            # Check for IsAllowableToLookDown in questions
            if ($attemptRes.questions) {
                Write-Host "`n=== IsAllowableToLookDown CHECK ==="
                $q = $attemptRes.questions[0]
                if ($null -ne $q.isAllowableToLookDown -or $null -ne $q.IsAllowableToLookDown) {
                    Write-Host "FOUND! isAllowableToLookDown is present on questions."
                    Write-Host "First question value: $($q.isAllowableToLookDown)$($q.IsAllowableToLookDown)"
                } else {
                    Write-Host "NOT FOUND: isAllowableToLookDown is NOT present on the question objects."
                    Write-Host "Available question properties: $(($q | Get-Member -MemberType NoteProperty).Name -join ', ')"
                }
            }
            break
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            Write-Host "Quiz code '$code' -> HTTP $status : $($_.ErrorDetails.Message)"
        }
    }
} catch {
    Write-Host "=== LOGIN FAILED ==="
    Write-Host $_.Exception.Message
    Write-Host $_.ErrorDetails.Message
}
