import {useLocation, Link, Outlet} from 'react-router-dom';
import {ReactComponent as Logo} from '@eda/assets/logo.svg';
import {ReactComponent as Dashboard} from '@eda/assets/dashboard.svg';
import {ReactComponent as Statistics} from '@eda/assets/statistic.svg';

import classNames from 'classnames';
import styles from './Root.module.css';
import React from "react";
import {SearchInput} from "@eda/ui/SearchInput/SearchInput.tsx";

export const Root = () => (
    <>
        <div className={styles.body}>
            <Aside/>
            <div className={styles.content}>
                <Header/>
                <main className={styles.main}>
                    <Outlet/>
                </main>
            </div>
        </div>
    </>
);

const navigationOptions = [
    {title: 'Dashboard', icon: Dashboard, path: '/'},
    {title: 'Statistics', icon: Statistics, path: '/statistics'},
];

function Aside({}) {
    const location = useLocation();

    return (
        <aside className={styles.aside}>
            <Link to='/' className={styles.logo}>
                <Logo/>
            </Link>
            <div className={styles.divider}/>
            <nav>
                <ul className={styles.navigation}>
                    {navigationOptions.map(option => (
                        <li key={option.title}>
                            <Link to={option.path} className={classNames(
                                styles.link,
                                {
                                    [styles.__active]: location.pathname === option.path,
                                },
                            )}>
                                {React.createElement(option.icon)}
                            </Link>
                        </li>
                    ))}
                </ul>
            </nav>
        </aside>
    );
}

function getTitle(curPath: string) {
    const option = navigationOptions.find(option => option.path === curPath);
    return option ? option.title : "EDA";
}

function Header() {
    const location = useLocation();

    const title = React.useMemo(
        () => getTitle(location.pathname),
        [location]
    );

    return (
        <header className={styles.header}>
            <h1 className={styles.header_text}>{title}</h1>
            <SearchInput/>
        </header>
    );
}
