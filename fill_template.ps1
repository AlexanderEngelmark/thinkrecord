$crate = Read-Host "To fill the template, tell me your egui project crate name: "
$name = Read-Host "To fill the template, tell me your name (for author in Cargo.toml): "
$email = Read-Host "To fill the template, tell me your e-mail address (also for Cargo.toml): "

Write-Host "Patching files..."

(Get-Content "Cargo.toml") -replace "ThinkRecord", $crate | Set-Content "Cargo.toml"
(Get-Content "src\main.rs") -replace "ThinkRecord", $crate | Set-Content "src\main.rs"
(Get-Content "index.html") -replace "ThinkRecord", $crate -replace "eframe_template", $crate | Set-Content "index.html"
(Get-Content "assets\sw.js") -replace "ThinkRecord", $crate | Set-Content "assets\sw.js"
(Get-Content "Cargo.toml") -replace "Alexander Engelmark", $name -replace "aleen199@student.liu.se", $email | Set-Content "Cargo.toml"

Write-Host "Done."
