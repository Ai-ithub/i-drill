/**
 * Language Switcher Component
 */
import { useI18n } from '../../i18n';
import { Globe } from 'lucide-react';

export function LanguageSwitcher() {
  const { language, setLanguage, isRTL } = useI18n();

  const toggleLanguage = () => {
    setLanguage(language === 'fa' ? 'en' : 'fa');
  };

  return (
    <button
      onClick={toggleLanguage}
      className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      title={language === 'fa' ? 'Switch to English' : 'تغییر به فارسی'}
      aria-label="Switch language"
    >
      <Globe className="w-5 h-5" />
      <span className="font-medium">{language === 'fa' ? 'EN' : 'FA'}</span>
    </button>
  );
}

