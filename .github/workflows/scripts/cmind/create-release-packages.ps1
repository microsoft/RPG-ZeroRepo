#!/usr/bin/env pwsh
#requires -Version 7.0

<#
.SYNOPSIS
    Build Spec Kit template release archives for each supported AI assistant and script type.

.DESCRIPTION
    create-release-packages.ps1 (workflow-local)
    Build Spec Kit template release archives for each supported AI assistant and script type.
    
.PARAMETER Version
    Version string with leading 'v' (e.g., v0.2.0)

.PARAMETER Agents
    Comma or space separated subset of agents to build (default: all)
    Valid agents: copilot, claude, gemini, cursor-agent, qwen, opencode, auggie, codex, codebuddy, qoder, amp

.PARAMETER Scripts
    Comma or space separated subset of script types to build (default: both)
    Valid scripts: sh, ps

.EXAMPLE
    .\create-release-packages.ps1 -Version v0.2.0

.EXAMPLE
    .\create-release-packages.ps1 -Version v0.2.0 -Agents claude,copilot -Scripts sh

.EXAMPLE
    .\create-release-packages.ps1 -Version v0.2.0 -Agents claude -Scripts ps
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Version,
    
    [Parameter(Mandatory=$false)]
    [string]$Agents = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Scripts = ""
)

$ErrorActionPreference = "Stop"

if ($Version -notmatch '^v\d+\.\d+\.\d+(-.+)?$') {
    Write-Error "Version must look like v0.0.0 or v0.0.0-dev.1"
    exit 1
}

$RepoRoot = if ($env:GITHUB_WORKSPACE) { $env:GITHUB_WORKSPACE } else { (git rev-parse --show-toplevel).Trim() }
$ProjectDir = if ($env:PROJECT_DIR) { $env:PROJECT_DIR } else { "CoderMind" }
$ProjectRoot = Join-Path $RepoRoot $ProjectDir
if (-not (Test-Path $ProjectRoot)) {
    Write-Error "CoderMind project directory not found: $ProjectRoot"
    exit 1
}
Set-Location $ProjectRoot

Write-Host "Building release packages for $Version from $ProjectRoot"

$GenReleasesDir = Join-Path $ProjectRoot ".genreleases"
if (Test-Path $GenReleasesDir) {
    Remove-Item -Path $GenReleasesDir -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $GenReleasesDir -Force | Out-Null

function Rewrite-Paths {
    param([string]$Content)

    $Content = $Content -replace '(/?)\bmemory/', '.cmind/memory/'
    $Content = $Content -replace '(/?)\bscripts/', '.cmind/scripts/'
    $Content = $Content -replace '(/?)\btemplates/', '.cmind/templates/'
    $Content = $Content -replace '(/?)\butils/', '.cmind/utils/'
    return $Content
}

function Generate-Commands {
    param(
        [string]$Extension,
        [string]$OutputDir
    )
    
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    
    $templates = Get-ChildItem -Path "templates/commands/*.md" -File -ErrorAction SilentlyContinue
    
    foreach ($template in $templates) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($template.Name)
        
        # Read file content and normalize line endings
        $body = (Get-Content -Path $template.FullName -Raw) -replace "`r`n", "`n"
        
        # Extract description from YAML frontmatter (for toml format)
        $description = ""
        if ($body -match '(?m)^description:\s*(.+)$') {
            $description = $matches[1]
        }
        
        # Rewrite paths for .cmind structure
        $body = Rewrite-Paths -Content $body
        
        # Generate output file based on extension
        $outputFile = Join-Path $OutputDir "cmind.$name.$Extension"
        
        switch ($Extension) {
            'toml' {
                $body = $body -replace '\\', '\\'
                $output = "description = `"$description`"`n`nprompt = `"`"`"`n$body`n`"`"`""
                Set-Content -Path $outputFile -Value $output -NoNewline
            }
            'md' {
                Set-Content -Path $outputFile -Value $body -NoNewline
            }
            'agent.md' {
                Set-Content -Path $outputFile -Value $body -NoNewline
            }
        }
    }
}

function Generate-CopilotPrompts {
    param(
        [string]$AgentsDir,
        [string]$PromptsDir
    )
    
    New-Item -ItemType Directory -Path $PromptsDir -Force | Out-Null
    
    $agentFiles = Get-ChildItem -Path "$AgentsDir/cmind.*.agent.md" -File -ErrorAction SilentlyContinue
    
    foreach ($agentFile in $agentFiles) {
        $basename = $agentFile.Name -replace '\.agent\.md$', ''
        $promptFile = Join-Path $PromptsDir "$basename.prompt.md"
        
        $content = @"
---
agent: $basename
---
"@
        Set-Content -Path $promptFile -Value $content
    }
}

function Build-Variant {
    param(
        [string]$Agent,
        [string]$Script
    )
    
    $baseDir = Join-Path $GenReleasesDir "sdd-${Agent}-package-${Script}"
    Write-Host "Building $Agent ($Script) package..."
    New-Item -ItemType Directory -Path $baseDir -Force | Out-Null
    
    # Copy base structure but filter scripts by variant
    $specDir = Join-Path $baseDir ".cmind"
    New-Item -ItemType Directory -Path $specDir -Force | Out-Null
    
    if (Test-Path "pyproject.toml") {
        Copy-Item -Path "pyproject.toml" -Destination (Join-Path $specDir "pyproject.toml") -Force
        Write-Host "Copied pyproject.toml -> .cmind"
    }

    # Copy memory directory
    if (Test-Path "memory") {
        Copy-Item -Path "memory" -Destination $specDir -Recurse -Force
        Write-Host "Copied memory -> .cmind"
    }
    
    # Only copy the relevant script variant directory
    if (Test-Path "scripts") {
        $scriptsDestDir = Join-Path $specDir "scripts"
        New-Item -ItemType Directory -Path $scriptsDestDir -Force | Out-Null
        
        switch ($Script) {
            'sh' {
                if (Test-Path "scripts/bash") {
                    Copy-Item -Path "scripts/bash" -Destination $scriptsDestDir -Recurse -Force
                    Write-Host "Copied scripts/bash -> .cmind/scripts"
                }
            }
            'ps' {
                if (Test-Path "scripts/powershell") {
                    Copy-Item -Path "scripts/powershell" -Destination $scriptsDestDir -Recurse -Force
                    Write-Host "Copied scripts/powershell -> .cmind/scripts"
                }
            }
        }
        
        # Copy any script files that aren't in variant-specific directories
        Get-ChildItem -Path "scripts" -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination $scriptsDestDir -Force
        }
        
        # Copy all subdirectories under scripts
        Get-ChildItem -Path "scripts" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination $scriptsDestDir -Recurse -Force
        }
    }
    
    # Copy templates (excluding commands directory and vscode-settings.json)
    if (Test-Path "templates") {
        $templatesDestDir = Join-Path $specDir "templates"
        New-Item -ItemType Directory -Path $templatesDestDir -Force | Out-Null
        
        Get-ChildItem -Path "templates" -Recurse -File | Where-Object {
            $_.FullName -notmatch 'templates[/\\]commands[/\\]' -and $_.Name -ne 'vscode-settings.json'
        } | ForEach-Object {
            $relativePath = $_.FullName.Substring((Resolve-Path "templates").Path.Length + 1)
            $destFile = Join-Path $templatesDestDir $relativePath
            $destFileDir = Split-Path $destFile -Parent
            New-Item -ItemType Directory -Path $destFileDir -Force | Out-Null
            Copy-Item -Path $_.FullName -Destination $destFile -Force
        }
        Write-Host "Copied templates -> .cmind/templates"
    }
    
    # Copy utils directory
    if (Test-Path "utils") {
        Copy-Item -Path "utils" -Destination $specDir -Recurse -Force
        Write-Host "Copied utils -> .cmind/utils"
    }
    
    # Replace <AI_CLI_CMD> placeholder in copied scripts with the actual CLI command name
    if (Test-Path (Join-Path $specDir "scripts")) {
        $agentName = ""
        switch ($Agent) {
            'copilot' { $agentName = "copilot" }
            'claude' { $agentName = "claude" }
            'gemini' { $agentName = "gemini -p" }
            'qwen' { $agentName = "qwen -p" }
            'cursor-agent' { $agentName = "agent -p" }
            'auggie' { $agentName = "augment -p" }
            'codex' { $agentName = "codex exec" }
            'codebuddy' { $agentName = "codebuddy -p" }
            'qoder' { $agentName = "qodercli -p" }
            'opencode' { $agentName = "opencode run" }
            'amp' { $agentName = "amp --execute" }
            default { $agentName = "" }
        }
        
        # Only perform replacement if agentName is set
        if (-not [string]::IsNullOrEmpty($agentName)) {
            $scriptsPath = Join-Path $specDir "scripts"
            Get-ChildItem -Path $scriptsPath -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
                $content = Get-Content -Path $_.FullName -Raw -ErrorAction SilentlyContinue
                if ($null -ne $content) {
                    $newContent = $content -replace '<AI_CLI_CMD>', $agentName
                    if ($content -ne $newContent) {
                        Set-Content -Path $_.FullName -Value $newContent -NoNewline
                    }
                }
            }
            Write-Host "Replaced <AI_CLI_CMD> with '$agentName' in scripts"
        } else {
            Write-Host "Skipped <AI_CLI_CMD> replacement (no CLI command for $Agent)"
        }
    }
    
    # Generate agent-specific command files
    switch ($Agent) {
        'claude' {
            $cmdDir = Join-Path $baseDir ".claude/commands"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
            $settingsContent = @'
{
  "permissions": {
    "allow": [
      "Write",
      "Edit",
      "Read",
      "Glob",
      "Grep",
      "Bash",
      "WebFetch"
    ],
    "deny": [
      "WebSearch"
    ]
  }
}
'@
            Set-Content -Path (Join-Path $baseDir ".claude/settings.json") -Value $settingsContent -NoNewline
        }
        'gemini' {
            $cmdDir = Join-Path $baseDir ".gemini/commands"
            Generate-Commands -Extension 'toml' -OutputDir $cmdDir
            if (Test-Path "agent_templates/gemini/GEMINI.md") {
                Copy-Item -Path "agent_templates/gemini/GEMINI.md" -Destination (Join-Path $baseDir "GEMINI.md")
            }
        }
        'copilot' {
            $agentsDir = Join-Path $baseDir ".github/agents"
            Generate-Commands -Extension 'agent.md' -OutputDir $agentsDir
            
            # Generate companion prompt files
            $promptsDir = Join-Path $baseDir ".github/prompts"
            Generate-CopilotPrompts -AgentsDir $agentsDir -PromptsDir $promptsDir
            
            # Create VS Code workspace settings
            $vscodeDir = Join-Path $baseDir ".vscode"
            New-Item -ItemType Directory -Path $vscodeDir -Force | Out-Null
            if (Test-Path "templates/vscode-settings.json") {
                Copy-Item -Path "templates/vscode-settings.json" -Destination (Join-Path $vscodeDir "settings.json")
            }
        }
        'cursor-agent' {
            $cmdDir = Join-Path $baseDir ".cursor/commands"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
        }
        'qwen' {
            $cmdDir = Join-Path $baseDir ".qwen/commands"
            Generate-Commands -Extension 'toml' -OutputDir $cmdDir
            if (Test-Path "agent_templates/qwen/QWEN.md") {
                Copy-Item -Path "agent_templates/qwen/QWEN.md" -Destination (Join-Path $baseDir "QWEN.md")
            }
        }
        'auggie' {
            $cmdDir = Join-Path $baseDir ".augment/commands"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
        }
        'codex' {
            $cmdDir = Join-Path $baseDir ".codex/prompts"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
        }
        'codebuddy' {
            $cmdDir = Join-Path $baseDir ".codebuddy/commands"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
        }
        'qoder' {
            $cmdDir = Join-Path $baseDir ".qoder/commands"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
        }
        'opencode' {
            $cmdDir = Join-Path $baseDir ".opencode/command"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
        }
        'amp' {
            $cmdDir = Join-Path $baseDir ".agents/commands"
            Generate-Commands -Extension 'md' -OutputDir $cmdDir
        }
    }
    
    # Create zip archive
    $zipFile = Join-Path $GenReleasesDir "cmind-template-${Agent}-${Script}-${Version}.zip"
    Compress-Archive -Path "$baseDir/*" -DestinationPath $zipFile -Force
    Write-Host "Created $zipFile"
}

# Define all agents and scripts
$AllAgents = @('copilot', 'claude', 'gemini', 'cursor-agent', 'qwen', 'opencode', 'auggie', 'codex', 'codebuddy', 'qoder', 'amp')
$AllScripts = @('sh', 'ps')

function Normalize-List {
    param([string]$Input)
    
    if ([string]::IsNullOrEmpty($Input)) {
        return @()
    }
    
    # Split by comma or space and remove duplicates while preserving order
    $items = $Input -split '[,\s]+' | Where-Object { $_ } | Select-Object -Unique
    return $items
}

function Validate-Subset {
    param(
        [string]$Type,
        [string[]]$Allowed,
        [string[]]$Items
    )
    
    $ok = $true
    foreach ($item in $Items) {
        if ($item -notin $Allowed) {
            Write-Error "Unknown $Type '$item' (allowed: $($Allowed -join ', '))"
            $ok = $false
        }
    }
    return $ok
}

# Determine agent list
if (-not [string]::IsNullOrEmpty($Agents)) {
    $AgentList = Normalize-List -Input $Agents
    if (-not (Validate-Subset -Type 'agent' -Allowed $AllAgents -Items $AgentList)) {
        exit 1
    }
} else {
    $AgentList = $AllAgents
}

# Determine script list
if (-not [string]::IsNullOrEmpty($Scripts)) {
    $ScriptList = Normalize-List -Input $Scripts
    if (-not (Validate-Subset -Type 'script' -Allowed $AllScripts -Items $ScriptList)) {
        exit 1
    }
} else {
    $ScriptList = $AllScripts
}

Write-Host "Agents: $($AgentList -join ', ')"
Write-Host "Scripts: $($ScriptList -join ', ')"

# Build all variants
foreach ($agent in $AgentList) {
    foreach ($script in $ScriptList) {
        Build-Variant -Agent $agent -Script $script
    }
}

Write-Host "`nArchives in ${GenReleasesDir}:"
Get-ChildItem -Path $GenReleasesDir -Filter "cmind-template-*-${Version}.zip" | ForEach-Object {
    Write-Host "  $($_.Name)"
}