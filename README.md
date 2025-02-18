# git-wiki

The git-wiki project is composed by 3 different repository:

- [git-wiki-theme](https://github.com/Drassil/git-wiki-theme): This is the repository of the theme that implements the wiki functionalities. You would have not fork it unless you need to send a Pull Request or create your wiki project from scratch.

- [git-wiki-skeleton](https://github.com/Drassil/git-wiki-skeleton): This is the repo that you should fork or use as a template. It uses the [jekyll remote theme](https://github.com/benbalter/jekyll-remote-theme) functionality that allows you to create your own wiki based on git-wiki-theme. By using the remote functionality you can automatically keep your wiki always updated with latest features from the **git-wiki-theme**, but you can also fully customize it. 

- [git-wiki](https://github.com/Drassil/git-wiki): This is the documentation repository and website of the **git-wiki-theme** project. You would have not fork it unless you want to contribute to the git-wiki project documentation.

## Features 

* Improvements in the **cooperative** aspect: forks, pull requests and roles.
* You can **customize your wiki** as you want with stylesheets and even changing the layout (see customization section below).
* **No databases!** Only static files that can be downloaded in a few seconds.
* **Blazing fast** and free thankfully to GitHub/Gitlab Pages and Jekyll Server Side Generation process!
* **Markdown and HTML** mixed together!
* **Multiple free search engines!** on a static site!
* **History, revision comparison** and everything you need from a wiki platform.
* You can **edit your pages** with the standard git editor, prose.io (integrated) or any kind of editor you prefer.
* Non-existent wiki page links are "[red](red.md)", you can **click on them to automatically create a new page**!
* [External links](http://example.com) get the right icon automatically.
* **Component system with hooks** that allows you to completely customize your wiki UI (see customization section below).
* Some **nice internal themes** to change your entire wiki UI with 1 simple configuration (see customization section below).
* Integrated **Blogging** feature thanks to Jekyll!
* Automatically generated **TOC**!
* You can download the entire wiki for **offline** usage and even navigate directly using a Markdown reader!
