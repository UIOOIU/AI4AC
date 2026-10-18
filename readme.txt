推荐（智能式）：直接安装TRAE，并把项目直接导入到TRAE中（文件-->打开文件夹），然后由TRAE自动安装所有依赖环境。

----- 以下是windows环境中环境准备
windows系统中安装conda运行环境
1) 下载正确miniconda
访问Miniconda — Anaconda documentation [https://docs.anaconda.com/miniconda/]  (从Latest Miniconda installer links里选择正确版本)
或直接从repo[https://repo.anaconda.com/miniconda/]里选取最新的正确版本

2) 安装至目录 (比如 d:/program/mincoda3)
这个过程都是在软件安装窗口中操作。过程中需要指定软件安装目录；勾选“Add Mincoda3 to my PATH enviroment variable”及"Register Mincoda3 as my ..."。

3）命令行安装python包 (注：此处直接安装在缺省环境中，如要建虚拟环境可自行baidu)
进入windows命令行? 
法1：windows的开始->搜索中输入“prompt”->选择“Anaconda Prompt(Mincoda3); 法2: windows资源管理器，某个目录中鼠标右键选择“在终端中打开”; 法3: WIN+R 的弹出中输入"CMD";

下面的">"表示在命令情况下的提示符(copy命令时只需要">"后面的部分)
> conda install pandas numpy scikit-learn matplotlib seaborn
> conda install jupyter

-----
以上安装过程也可参考https://blog.csdn.net/weixin_57972634/article/details/142107300