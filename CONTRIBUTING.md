# Contributing

I would like to give you all some context.

I will talk to John Mason, but it’s not about you or your PR, John Mason, sorry for hijacking this opportunity though! I think it’s fair that you should know it though. I apologize for the length.

## V0 flashbacks: Saas was not… OpenHands’ best friend

We will soon hit 2 years in OpenHands’ existence. Most part of it was V0. In V0, so it happened, a little many times I found myself with some AI researcher or other open source contributors, debugging their reports or PRs, me saying, hey that should work, look at the code there, and they were like, nope not anymore. Digging into it, we found that it broke, alright, and looking more into it, it turned out the breakage came from the Saas. And the other one, came from the Saas. And the other.

Some were easy fixes, some were close to impossible (without breaking the presumed goals of that code, anyway).

I lost count how many times implementing stuff for the Saas was traceable as the origin why the agent loop got messed up, or a bunch of asyncs were not playing nice with other use cases than the Saas server, or the agent controller no longer closed cleanly, breaking instances in some non-SWE bench evals.

Yep, over and over.

And to be clear, many times, far from all, but many times, people out there, people not in AH, people not working on the Saas, kept fixing them.

We figured there were deep architectural problems, attempting to support too much in a monolith, and rebuilt.

## OpenHands SDK

We are in a different place today. One reason why the OpenHands SDK exists, as a separate repo and deeply rebuilt, is exactly that kind of thing.

It’s what OpenHands maybe should have been, so that an application with the complexity of the Saas can be built on it and work well.

It's a deep architectural rebuild precisely to isolate the agent loop from the client apps flows interfering with it. They should go through standard interfaces and common flows. Not weird shortcuts.

That’s also why some of us care about some code design guidelines, and yes, sometimes may want to mull over things a bit more.

As Xingyao said - we’ve formed some habits, informal-ish processes, on how to accept significant changes in, and put them through the same workflow and testing as everyone else’s PRs.

(Side note: this implies testing on the Open Source project(s), not only on the Saas. Validating Saas-only risks breaking things you don’t see, because of env vars/different workspaces or whatnot.)

Please let me put this way: the OpenHands SDK is an Open Source project, on which client applications are built.

- the CLI is a client application built on the SDK.
- OH app-server is a client application built on the SDK.
- the Saas is a client application built on the SDK.

And so are others.

I believe we’re hoping to see many more.

(Tiny aside: The What and The How.)

The SDK needs to work for them and their users, too, not just for the Saas and enterprises. That, I admit, matters a lot to me.

The Saas is maybe the best client app in some sense - as a test of the SDK - because of its size and complexity. But the SDK is not an appendage of the Saas.

I’m sure I’m not saying anything new if I say, that does mean how we do things to support the use cases you wish, matters too. Not only the what.

I do think we are aligned on the what.

It’s just some stuff about the how, that is not totally clear to me.

Sorry for flashbacks, let’s move forward.

Also, apologies for the length, and for being dense, questioning things that maybe should be simple. But we’re figuring it out.

## Back to the PRs

I believe we have a lot of work ahead of us on context management.

Your PRs on this are actually in an interesting place at the intersection of context management, extensibility (the lack of which is why things broke in v0!) and ease of use.

I’m confident we can find a good solution! Something we like and it’s aligned with CC, with OH CLI, and with what is to come.
