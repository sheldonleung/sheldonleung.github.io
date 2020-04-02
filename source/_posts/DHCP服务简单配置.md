---
title: DHCP服务简单配置
date: 2019-09-12 00:26:56
categories: 踩的坑
tags: DHCP
---

&nbsp;&nbsp;&nbsp;&nbsp;这个学期的一门课叫“信息系统运行与维护”，第一次实践内容就是让我们配置DHCP服务，鉴于同学们(包括我)都碰到了不少困难，在这里我就简单归纳了一下DHCP的一些配置。

&nbsp;&nbsp;&nbsp;&nbsp;**我这里没有配置DNS，如果有需要的同学可以另外找设置方法。**

<!--more-->

# 1 确定自己的Linux版本

&nbsp;&nbsp;&nbsp;&nbsp;因为各个Linux发行版之间的一些差别，一些配置文件的路径会有所不同，所以首先确定自己的Linux发行版是Redhat系列的还是Debian系列的，而且还要明确知道自己用的是哪个版本，因为同一个发行版，不同版本之间也会有差异(比如Ubuntu18.04)。在这里我主要列出Redhat和Debian系列的，原因是这两个系列的发行版用的相对比较多。

# 2 安装DHCP

Redhat的安装：`rpm -vih /dhcp的rpm包所在的路径/dhcp-x.x.x.rpm`

CentOS的安装：`yum install dhcp` 或者使用Redhat的安装方式**(Redhat系列的软件安装方式都是通用的)**

Debian系列的安装：`apt-get install isc-dhcp-server`

# 3 DHCP服务端配置

<font size=5>**在配置前，请将配置文件备份，日后出问题有后悔药吃！！！**</font>

## 3.1 修改DHCP配置文件

&nbsp;&nbsp;&nbsp;&nbsp;配置dhcp，使用命令`vim /etc/dhcp/dhcpd.conf`如果提示没有该文件，先将配置文件的一份模板复制一份`cp /usr/share/doc/dhcp-x.x.x/dhcpd.conf.sample /etc/dhcp/dhcpd.conf`，然后在进行配置。

&nbsp;&nbsp;&nbsp;&nbsp;配置前，请查看你本机的IP，子网掩码，广播地址等信息。命令：`ifconfig -a`

配置的内容如下：

```
subnet x.x.x.0 netmask 255.255.255.0 {  # 这里最好是根据你服务端的IP来设置，不然DHCP无法启动
  range x.x.x.100 x.x.x.200;  # IP范围(再次提醒，要根据你服务端的IP来设置！)
  option routers x.x.x.1;  # 网卡
  option broadcast-address x.x.x.x;  # 你机器上的广播地址
  option domain-name-servers 8.8.8.8;  # DNS服务器，我这里使用了谷歌的免费DNS
  option domain-name "Google's DNS";  # DNS名称
  default-lease-time 600;  # 默认超时时间
  max-lease-time 7200;  # 最大超时时间
}
```

&nbsp;&nbsp;&nbsp;&nbsp;配置完成之后保存退出，使用`dhcpd -t`检测DHCP配置有没有问题。

<font size=3>**如果没有问题接下来就是将你的机器设置成静态IP。**</font>

## 3.2 DHCP服务端设置静态IP

### 3.2.1 Redhat系列的设置

`vim /etc/sysconfig/network-scripts/ifcfg-eth0`，`ifcfg-eth0`这里要和你机器的网卡名称对应。

```
BOOTPROTO=static  # 这里将dhcp改成static
NAME=eth0  # 这里是你机器上网卡的名称
DEVICE=eth0  # 这里是你机器的网卡
UUID=6b9eed96-016c-4186-911b-ae33c43bf6ba
HWADDR=00:15:5D:38:01:16  # 你机器网卡的MAC地址
IPADDR=x.x.x.x  # 你机器的IP
BROADCAST=x.x.x.x  # 广播地址
NETMASK=255.255.255.0  # 子网掩码
GATEWAY=x.x.x.x  # 网关
```



### 3.2.2 Debian系列的设置

`vim /etc/network/interfaces`

```
auto lo
auto eth0  # 开机自动连接网络
iface lo inet loopback
allow-hotplug eth0
iface eth0 inet static  # static表示使用固定ip，dhcp表述使用动态ip
address x.x.x.x  # 设置ip地址
netmask x.x.x.x  # 设置子网掩码
gateway x.x.x.x  # 设置网关
broadcast x.x.x.x  # 广播地址
```

<font size=3>**注意！**</font>

&nbsp;&nbsp;&nbsp;&nbsp;从Ubuntu17.10开始放弃在`/etc/network/interfaces`配置IP，改成使用netplan的方式，所以设置静态IP的方式与Debian不同**(Ubuntu17.10之前的版本不受影响)**。`vim /etc/netplan/01-netcfg.yaml`配置文件的名字会不一样，根据机的实际情况打开。

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: no  # 关闭DHCP，即设置成静态
      addresses: [x.x.x.x/24]  # 这里填要设置的IP及掩码
```

### 3.2.3 重启DHCP服务

&nbsp;&nbsp;&nbsp;&nbsp;设置完之后保存并退出，Ubuntu18.04还要加`netplan apply`这一步使配置生效。

&nbsp;&nbsp;&nbsp;&nbsp;接下来重启network服务`systemctl network restart`，查看IP是否设置成功。

&nbsp;&nbsp;&nbsp;&nbsp;做好上面的步骤就可以重启DHCP服务了`systemctl restart dhcpd.service`，如果启动不了，查看系统日志`tail -30 /var/log/messages`，确定出错的地方再找解决问题的办法。

# 4 DHCP客户端配置

## 4.1 Redhat系列的配置

&nbsp;&nbsp;&nbsp;&nbsp;修改网卡配置文件`vim /etc/sysconfig/network-scripts/ifcfg-eth0`

```
DEVICE=eth0
BOOTPROTO=dhcp  # 改成dhcp
HWADDR=00:0C:29:B6:F0:F4
IPV6INIT=yes
ONBOOT=yes
TYPE=Ethernet
# UUID=de66d5f4-37fa-4cdb-8e05-527314d0da80
#注释掉UUID HWADDR IPADDR NETMASK GATEWAY
```

&nbsp;&nbsp;&nbsp;&nbsp;保存并退出

## 4.2 Debian系列的配置

&nbsp;&nbsp;&nbsp;&nbsp;修改网卡配置文件`vim /etc/network/interfaces`

```
auto eth0
iface eth0 inet dhcp
```

&nbsp;&nbsp;&nbsp;&nbsp;保存并退出

## 4.3 Ubuntu17.10及以上版本的配置

```yaml
network:
  verison: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: true  # 开启DHCP
      addresses: []  # 这里面是空
```

&nbsp;&nbsp;&nbsp;&nbsp;保存并退出，执行`netplan apply`使配置生效。

## 4.4 重启network

`systemctl restart network`重启之后查看网卡配置`ifconfig -a`，查看DNS信息`cat /etc/resolv.conf`(我这里就没有设置DNS，如果有特殊需要设置DNS的就另外找解决的办法啦)

# 5 问题总结

<font size=5>**服务器DHCP没有生效**</font>

&nbsp;&nbsp;&nbsp;&nbsp;问题描述：当客户端修改网卡配置文件的BOOTPROTO为dhcp，重启network后查看IP与创建的DHCP的IP范围不符合

&nbsp;&nbsp;&nbsp;&nbsp;解决办法：切到VMware客户端的窗口，上面的菜单栏：编辑->虚拟网络编辑器->更改设置，你所在的模式有个**使用本地DHCP服务将IP地址分配给虚拟机**选项勾掉。VirtualBox同样也能够关闭自带的DHCP服务。

