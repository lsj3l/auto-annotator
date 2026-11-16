-- Optional macOS launcher for a local checkout of Auto Annotator.
-- Open in Script Editor and export as an Application to use it from Finder.
use scripting additions

property projectFolderPath : ""

on run
	if projectFolderPath is "" then
		set chosenFolder to choose folder with prompt "Choose your Auto Annotator project folder"
		set projectFolderPath to POSIX path of chosenFolder
	end if
	try
		do shell script "/usr/bin/test -f " & quoted form of (projectFolderPath & "scripts/run.sh")
	on error
		set projectFolderPath to ""
		display dialog "Choose the project folder containing app.py and scripts/run.sh on the next launch." buttons {"OK"} default button "OK"
		return
	end try
	set logPath to (POSIX path of (path to temporary items)) & "auto-annotator-launch.log"
	set startCommand to "cd " & quoted form of projectFolderPath & " && if ! /usr/bin/curl -fsS --max-time 1 http://127.0.0.1:5001/api/config >/dev/null 2>&1; then nohup /bin/bash scripts/run.sh >> " & quoted form of logPath & " 2>&1 < /dev/null & fi"
	do shell script "/bin/zsh -lc " & quoted form of startCommand
	repeat 120 times
		try
			do shell script "/usr/bin/curl -fsS --max-time 1 http://127.0.0.1:5001/api/config >/dev/null 2>&1"
			open location "http://127.0.0.1:5001"
			return
		on error
			delay 1
		end try
	end repeat
	display dialog "Auto Annotator is still starting, or another app is using port 5001. Setup details are in " & logPath buttons {"OK"} default button "OK"
end run
