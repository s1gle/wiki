# git-wiki

The git-wiki project is composed by 3 different repository:

- [git-wiki-theme](https://github.com/Drassil/git-wiki-theme): This is the repository of the theme that implements the wiki functionalities. You would have not fork it unless you need to send a Pull Request or create your wiki project from scratch.

- [git-wiki-skeleton](https://github.com/Drassil/git-wiki-skeleton): This is the repo that you should fork or use as a template. It uses the [jekyll remote theme](https://github.com/benbalter/jekyll-remote-theme) functionality that allows you to create your own wiki based on git-wiki-theme. By using the remote functionality you can automatically keep your wiki always updated with latest features from the **git-wiki-theme**, but you can also fully customize it. 

- [git-wiki](https://github.com/Drassil/git-wiki): This is the documentation repository and website of the **git-wiki-theme** project. You would have not fork it unless you want to contribute to the git-wiki project documentation.

## Features 

* Улучшения в **кооперативном** аспекте: форки, запросы на включение и роли.
* Вы можете **настроить свою вики** по своему усмотрению, используя таблицы стилей и даже изменяя макет (см. раздел настройки ниже).
* **Никаких баз данных!** Только статические файлы, которые можно скачать за несколько секунд.
* **Невероятно быстро** и бесплатно благодаря GitHub/Gitlab Pages и процессу генерации серверной части Jekyll!
* **Markdown и HTML** смешаны вместе!
* **Несколько бесплатных поисковых систем!** на статическом сайте!
* **История, сравнение версий** и все, что вам нужно от вики-платформы.
* Вы можете **редактировать свои страницы** с помощью стандартного редактора git, prose.io (встроенного) или любого другого редактора, который вы предпочитаете.
* Несуществующие ссылки на вики-страницы — «[красные](red.md)», вы можете **щелкнуть по ним, чтобы автоматически создать новую страницу**!
* [Внешние ссылки](http://example.com) автоматически получают правильный значок.
* **Система компонентов с крючками**, позволяющая полностью настроить пользовательский интерфейс вики (см. раздел настройки ниже).
* Несколько **приятных внутренних тем**, позволяющих изменить весь пользовательский интерфейс вики с помощью одной простой настройки (см. раздел настройки ниже).
* Интегрированная функция **ведения блога** благодаря Jekyll!
* Автоматически генерируется **TOC**!
* Вы можете загрузить всю вики для **офлайн** использования и даже напрямую перемещаться по ней с помощью программы чтения Markdown!
