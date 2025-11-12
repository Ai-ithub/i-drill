/**
 * Internationalization (i18n) Configuration
 * Supports Persian (Farsi) and English
 */
import { createContext, useContext, useState, ReactNode } from 'react';

export type Language = 'fa' | 'en';

export interface Translations {
  [key: string]: string | Translations;
}

const translations: Record<Language, Translations> = {
  fa: {
    common: {
      save: 'ذخیره',
      cancel: 'لغو',
      delete: 'حذف',
      edit: 'ویرایش',
      create: 'ایجاد',
      search: 'جستجو',
      filter: 'فیلتر',
      export: 'خروجی',
      import: 'ورودی',
      loading: 'در حال بارگذاری...',
      error: 'خطا',
      success: 'موفق',
      warning: 'هشدار',
      info: 'اطلاعات',
    },
    dashboard: {
      title: 'داشبورد',
      overview: 'نمای کلی',
      realTime: 'زمان واقعی',
      historical: 'تاریخی',
    },
    sensor: {
      title: 'داده‌های سنسور',
      rigId: 'شناسه Rig',
      depth: 'عمق',
      wob: 'وزن روی مته',
      rpm: 'دور بر دقیقه',
      torque: 'گشتاور',
      pressure: 'فشار',
      temperature: 'دما',
    },
    predictions: {
      title: 'پیش‌بینی‌ها',
      rul: 'پیش‌بینی RUL',
      anomaly: 'تشخیص آنومالی',
      model: 'مدل',
      selectModel: 'انتخاب مدل',
    },
    maintenance: {
      title: 'تعمیر و نگهداری',
      alerts: 'هشدارها',
      schedules: 'برنامه‌ها',
      severity: 'شدت',
      critical: 'بحرانی',
      high: 'بالا',
      medium: 'متوسط',
      low: 'پایین',
    },
    auth: {
      login: 'ورود',
      logout: 'خروج',
      username: 'نام کاربری',
      password: 'رمز عبور',
      rememberMe: 'مرا به خاطر بسپار',
    },
  },
  en: {
    common: {
      save: 'Save',
      cancel: 'Cancel',
      delete: 'Delete',
      edit: 'Edit',
      create: 'Create',
      search: 'Search',
      filter: 'Filter',
      export: 'Export',
      import: 'Import',
      loading: 'Loading...',
      error: 'Error',
      success: 'Success',
      warning: 'Warning',
      info: 'Info',
    },
    dashboard: {
      title: 'Dashboard',
      overview: 'Overview',
      realTime: 'Real-time',
      historical: 'Historical',
    },
    sensor: {
      title: 'Sensor Data',
      rigId: 'Rig ID',
      depth: 'Depth',
      wob: 'Weight on Bit',
      rpm: 'RPM',
      torque: 'Torque',
      pressure: 'Pressure',
      temperature: 'Temperature',
    },
    predictions: {
      title: 'Predictions',
      rul: 'RUL Prediction',
      anomaly: 'Anomaly Detection',
      model: 'Model',
      selectModel: 'Select Model',
    },
    maintenance: {
      title: 'Maintenance',
      alerts: 'Alerts',
      schedules: 'Schedules',
      severity: 'Severity',
      critical: 'Critical',
      high: 'High',
      medium: 'Medium',
      low: 'Low',
    },
    auth: {
      login: 'Login',
      logout: 'Logout',
      username: 'Username',
      password: 'Password',
      rememberMe: 'Remember Me',
    },
  },
};

interface I18nContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
  isRTL: boolean;
}

const I18nContext = createContext<I18nContextType | undefined>(undefined);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>(() => {
    // Get from localStorage or default to browser language
    const saved = localStorage.getItem('i18n-language') as Language;
    if (saved && (saved === 'fa' || saved === 'en')) {
      return saved;
    }
    const browserLang = navigator.language.split('-')[0];
    return browserLang === 'fa' ? 'fa' : 'en';
  });

  const t = (key: string): string => {
    const keys = key.split('.');
    let value: any = translations[language];
    
    for (const k of keys) {
      value = value?.[k];
      if (value === undefined) {
        console.warn(`Translation missing for key: ${key}`);
        return key;
      }
    }
    
    return typeof value === 'string' ? value : key;
  };

  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang);
    localStorage.setItem('i18n-language', lang);
    document.documentElement.dir = lang === 'fa' ? 'rtl' : 'ltr';
    document.documentElement.lang = lang;
  };

  // Set initial direction
  if (typeof document !== 'undefined') {
    document.documentElement.dir = language === 'fa' ? 'rtl' : 'ltr';
    document.documentElement.lang = language;
  }

  return (
    <I18nContext.Provider
      value={{
        language,
        setLanguage: handleSetLanguage,
        t,
        isRTL: language === 'fa',
      }}
    >
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (context === undefined) {
    throw new Error('useI18n must be used within I18nProvider');
  }
  return context;
}

