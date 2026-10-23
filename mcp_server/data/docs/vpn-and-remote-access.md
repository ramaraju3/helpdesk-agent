# VPN and Remote Access Runbook

## When you need the VPN
Internal tools such as the wiki, the finance system and the build servers are only reachable through the VPN. Email, chat and the cloud drive work without it.

## Choosing a gateway
The VPN client picks the nearest gateway automatically: US-East, EU-West or AP-South. If internal pages are slow, open the client settings and pick a gateway manually; a gateway on another continent adds noticeable delay.

## Working while travelling abroad
Tell IT at least one week before travelling to another country. Some countries block or restrict VPN traffic, and logins from an unexpected country trigger an automatic account lock unless travel was registered beforehand.

## Gateway outages
If nobody can reach internal sites, check the status page first. A gateway outage affecting many people is a high-severity incident and goes to the network on-call engineer.
